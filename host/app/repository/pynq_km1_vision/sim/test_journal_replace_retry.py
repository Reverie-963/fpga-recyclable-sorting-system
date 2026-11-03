import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from src.sorting_loop import Journal


class JournalReplaceTests(unittest.TestCase):
    def test_windows_transient_denial_retries_same_saved_bytes(self):
        for code in (5,32,33):
            with self.subTest(code=code),tempfile.TemporaryDirectory() as folder:
                journal=Journal(folder);journal.save('pending.json',{'old':True})
                original=Path(folder,'pending.json').read_bytes()
                import os
                replace=os.replace;calls=[]
                def denied_once(source,target):
                    calls.append(Path(source).read_bytes())
                    if len(calls)==1:
                        self.assertEqual(Path(target).read_bytes(),original)
                        exc=PermissionError('sharing');exc.winerror=code;raise exc
                    return replace(source,target)
                with patch('src.sorting_loop.os.replace',side_effect=denied_once),patch('src.sorting_loop.time.sleep') as sleep:
                    journal.save('pending.json',{'new':True})
                self.assertEqual(calls[0],calls[1]);sleep.assert_called_once_with(.05)
                self.assertEqual(journal.pending(),{'new':True})

    def test_persistent_failure_keeps_old_pending_and_raises(self):
        with tempfile.TemporaryDirectory() as folder:
            journal=Journal(folder);journal.save('pending.json',{'old':True})
            exc=PermissionError('denied');exc.winerror=5
            with patch('src.sorting_loop.os.replace',side_effect=exc) as replace,patch('src.sorting_loop.time.sleep'):
                with self.assertRaises(PermissionError):journal.save('pending.json',{'new':True})
            self.assertEqual(replace.call_count,10);self.assertEqual(journal.pending(),{'old':True})
            self.assertEqual(json.loads(Path(folder,'pending.json.tmp').read_text()),{'new':True})

    def test_other_errors_are_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('src.sorting_loop.os.replace',side_effect=PermissionError('ACL')) as replace,patch('src.sorting_loop.time.sleep') as sleep:
                with self.assertRaises(PermissionError):Journal(folder).save('pending.json',{})
            self.assertEqual(replace.call_count,1);sleep.assert_not_called()
