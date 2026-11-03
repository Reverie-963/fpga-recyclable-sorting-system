import copy
import unittest
from src.prompted_grasp_override import requested_grasp_planes
from sim.test_sorting_unified import test_config,setup_policy


class GraspOverrideTests(unittest.TestCase):
    def test_explicit_trial_only_changes_requested_plane_and_chain_endpoint(self):
        c=test_config();p=setup_policy(c);before=copy.deepcopy(c);old=copy.deepcopy(p)
        result,policy,audit=requested_grasp_planes(c,p,{'requested_grasp_z_mm':{'plastic':8},
            'grasp_override_operator_statement':'operator requested 8mm'}, {'basis':'operator_supervised_trial'})
        self.assertEqual(c,before);self.assertEqual(p,old)
        self.assertEqual(result['profiles']['plastic']['grasp_z_mm'],8)
        self.assertTrue(all(chain[-1][0]==8 for chain in policy['models']['plastic']['source_chains']))
        self.assertEqual(result['profiles']['metal'],c['profiles']['metal'])
        self.assertEqual(result['known'],c['known'])
        self.assertFalse(audit['physical_validation_completed'])
        self.assertTrue(audit['original_qualification_preserved'])

    def test_missing_authority_invalid_height_and_mismatched_chain_are_rejected(self):
        c=test_config();p=setup_policy(c)
        options={'requested_grasp_z_mm':{'plastic':8},'grasp_override_operator_statement':'requested'}
        with self.assertRaisesRegex(ValueError,'supervised_trial'):requested_grasp_planes(c,p,options,{})
        with self.assertRaisesRegex(ValueError,'requested_grasp_z'):
            requested_grasp_planes(c,p,dict(options,requested_grasp_z_mm={'plastic':-1}),{'basis':'operator_supervised_trial'})
        p['models']['plastic']['source_chains'][0][-1][0]=2
        with self.assertRaisesRegex(ValueError,'chain_mismatch'):requested_grasp_planes(c,p,options,{'basis':'operator_supervised_trial'})


if __name__=='__main__':unittest.main()
