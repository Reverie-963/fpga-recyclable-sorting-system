import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).parents[1]))
from item_profile_store import ItemProfileStore


def profile():
    return dict(name='红色吸管',pose='平放',material='plastic',height_mm=10,width_mm=10,
        official_label='Plastic Straw',reference_model='a'*64,features=dict(rgb=[1,0,0],
        hist=[1]+[0]*11,aspect=20,area=.003),thumbnail='')


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.path=Path(self.temp.name)/'profiles.json';self.store=ItemProfileStore(self.path)

    def save(self,p=None,**values):
        return self.store.command(dict(action='save',expected_revision=self.store.read()['revision'],profile=p or profile(),**values))

    def test_persist_update_restart_delete_and_no_geometry_saved(self):
        first=self.save();row=first['profiles'][0]
        self.assertEqual(ItemProfileStore(self.path).read(),first)
        changed=profile();changed['height_mm']=11
        second=self.save(changed,id=row['id']);self.assertEqual(second['profiles'][0]['revision'],2)
        self.assertEqual(second['profiles'][0]['width_mm'],10)
        self.assertNotIn('mask_rle',row);self.assertNotIn('image_size',row)
        self.assertEqual(self.store.command(dict(action='delete',expected_revision=2,id=row['id']))['profiles'],[])

    def test_stale_other_browser_cannot_overwrite(self):
        self.save();before=self.path.read_bytes()
        with self.assertRaisesRegex(ValueError,'其他页面'):self.store.command(dict(action='save',expected_revision=0,profile=profile()))
        self.assertEqual(before,self.path.read_bytes())

    def test_invalid_dimensions_features_fields_and_thumbnail_leave_file_unchanged(self):
        self.save();before=self.path.read_bytes()
        for key,value in [('height_mm',True),('height_mm',0),('width_mm',float('nan')),('width_mm',151),
                          ('mask_rle',[1,2]),('thumbnail','data:image/svg+xml;base64,AAAA'),('reference_model','old')]:
            p=profile();p[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):self.save(p)
        self.assertEqual(before,self.path.read_bytes())

    def test_corrupt_library_never_silently_reset(self):
        self.path.write_text('{broken',encoding='utf-8')
        with self.assertRaises(ValueError):self.save()
        self.assertEqual(self.path.read_text(),'{broken')


if __name__=='__main__':unittest.main()
