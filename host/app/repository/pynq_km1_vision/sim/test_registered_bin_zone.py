"""Registered bin classification only; no camera, serial port or motion."""
import unittest
from types import SimpleNamespace
import numpy as np
from src.sorting_unified import registered_bin_zone,UnifiedController


class RegisteredBinZoneTests(unittest.TestCase):
    def setUp(self):
        self.mask=np.zeros((100,100),bool);self.mask[10:20,60:70]=True
        self.profiles={'metal':{'bin_roi':[60,12,80,40]}}
        self.reference={'metal':{'bin_roi':[60,10,80,40]}}
        self.source=[10,10,40,50]

    def zone(self,**kwargs):
        return registered_bin_zone(self.mask,self.profiles,[100,100],source_roi=kwargs.get('source',self.source),
            reference_profiles=kwargs.get('reference',self.reference))

    def test_projection_shift_can_use_bound_bin_outside_source_without_lowering_threshold(self):
        self.assertIsNone(registered_bin_zone(self.mask,self.profiles,[100,100]))
        self.assertEqual(self.zone(),'bin_metal')
        self.assertIsNone(self.zone(reference={'metal':{'bin_roi':[60,10,65,20]}}))

    def test_reference_cannot_hide_a_source_intrusion_or_ambiguous_bin(self):
        self.assertIsNone(self.zone(source=[60,10,65,40]))
        profiles=dict(self.profiles,plastic={'bin_roi':[0,80,10,90]})
        refs=dict(self.reference,plastic=self.reference['metal'])
        self.assertIsNone(registered_bin_zone(self.mask,profiles,[100,100],source_roi=self.source,reference_profiles=refs))

    def test_bin_stays_out_of_height_pool_but_its_raw_mask_is_retained(self):
        c=object.__new__(UnifiedController)
        c.c=dict(image_size=[100,100],source_roi=self.source,profiles=self.profiles,layout={'static_rois':[]})
        c.geometry=SimpleNamespace(original={'profiles':self.reference})
        c.policy={'detection_window_px':[0,0,100,100],'max_scene_instances':6}
        plastic=np.zeros_like(self.mask);plastic[20:30,20:30]=True
        objects=[dict(label='plastic',bbox_xyxy=[20,20,30,30]),dict(label='metal',bbox_xyxy=[60,10,70,20])]
        row=dict(image=np.zeros((100,100,3),np.uint8),observation={'instances':objects},masks=[plastic,self.mask])
        self.assertEqual([o['label'] for o,_ in c.source_objects(row)],['plastic'])
        self.assertEqual([o['label'] for o,_ in c.outside_objects(row)],['metal'])
        self.assertEqual(len(row['masks']),2)

    def test_fully_contained_current_bin_keeps_original_rule(self):
        self.profiles['metal']['bin_roi']=[60,10,80,40]
        self.assertEqual(registered_bin_zone(self.mask,self.profiles,[100,100]),'bin_metal')


if __name__=='__main__':unittest.main()
