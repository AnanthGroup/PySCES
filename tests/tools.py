import numpy as np
import io
import unittest

def parse_xyz_data(file_loc):
    n_atoms = None
    all_frames = []
    with open(file_loc) as file:
        for line in file:
            if n_atoms is None:
                n_atoms = int(line)
                next(file)
                frame_data = {'atoms': [], 'positions': []}
                for i in range(n_atoms):
                    sp = next(file).split()
                    coords = [float(x) for x in sp[1:4]]
                    frame_data['atoms'].append(sp[0])
                    frame_data['positions'].append(coords)
                frame_data['atoms'] = np.array(frame_data['atoms'])
                frame_data['positions'] = np.array(frame_data['positions'])
                n_atoms = None
                all_frames.append(frame_data)


    return all_frames

def assert_dictionary(testcase: unittest.TestCase, dict_ref, dict_tst, atol=1e-6, rtol=1e-6,):
    for key, value in dict_ref.items():
        if key not in dict_tst:
            raise ValueError(f'key "{key}" not in test dictionary')
        msg = f'value for key "{key}" not equal between both dicts'
        if isinstance(value, dict):
            assert_dictionary(dict_ref[key], dict_tst[key])
        elif isinstance(value, str) or isinstance(value, int):
            testcase.assertEqual(value, dict_tst[key], msg=msg)
        elif isinstance(value, float):
            testcase.assertAlmostEqual(value, dict_tst[key], places=8, msg=msg)
        elif isinstance(value, np.ndarray):
            np.testing.assert_allclose(value, dict_tst[key], atol=atol, rtol=rtol, err_msg=msg)
        elif isinstance(value, list):
            array_ref = np.array(value)
            array_tst = np.array(dict_tst[key])
            np.testing.assert_allclose(array_ref, array_tst, atol=atol, rtol=rtol, err_msg=msg)
        