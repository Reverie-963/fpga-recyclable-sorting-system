from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]))
from activate_board_console import selected_process,PROJECT


class SelectionTests(unittest.TestCase):
    def test_only_exact_project_console_modules_and_ports_selected(self):
        good=dict(cwd=str(PROJECT),cmdline=['python','-m','src.sorting_prompted_console','--port','8810'])
        self.assertTrue(selected_process(good,{8810,8812}))
        for info in (dict(good,cwd=str(PROJECT.parent)),
                     dict(good,cmdline=['python','-m','src.sorting_monitor_server','--port','8810']),
                     dict(good,cmdline=['python','-m','src.sorting_prompted_console','--port','8811']),
                     dict(good,cmdline=['python','-c','src.sorting_prompted_console','--port','8810'])):
            self.assertFalse(selected_process(info,{8810,8812}))


if __name__=='__main__':unittest.main()
