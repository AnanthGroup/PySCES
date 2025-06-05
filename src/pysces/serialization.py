from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .qcRunners import TCRunner
    from .interpolation import SignFlipper

import json
import numpy as np
import importlib
import base64
from types import ModuleType
import pickle
from itertools import zip_longest

# from .interpolation import SignFlipper
# from .qcRunners import TCRunner

def is_json_compatible(obj):
    """Check if an object is JSON-compatible (str, int, float, bool, None, list, or dict)."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return True
    elif isinstance(obj, list):
        return all(is_json_compatible(item) for item in obj)
    elif isinstance(obj, dict):
        return all(isinstance(key, str) and is_json_compatible(value) for key, value in obj.items())
    return False

def serialize(obj):
    """Recursively serialize an object into a JSON-compatible structure."""

    
    if is_json_compatible(obj):
        return obj
    elif isinstance(obj, np.ndarray):
        # Custom handling for numpy arrays
        return NumpyArray_Serialize(obj)
    elif isinstance(obj, (list, dict)):
        # Recursively serialize lists and dictionaries
        if isinstance(obj, list):
            return [serialize(item) for item in obj]
        else:
            return {key: serialize(value) for key, value in obj.items()}
    elif isinstance(obj, tuple):
        return tuple_Serialize(obj)
    else:

        # Use custom serialization function if available on the class
        serialize_func_name = f"{obj.__class__.__name__}_Serialize"
        if serialize_func := getattr(obj.__class__, serialize_func_name, None):
            out_data = {'__type__': obj.__class__.__name__}
            out_data.update(serialize_func(obj))
            return out_data
        elif serialize_func := globals().get(serialize_func_name, None):
            out_data = {'__type__': obj.__class__.__name__}
            out_data.update(serialize_func(obj))
            return out_data

        # Use pickle for objects without a __dict__ attribute
        if not hasattr(obj, '__dict__'):
            try:
                out_data = {
                    '__type__': 'PickleSerialized',
                    'data': pickle.dumps(obj).hex()
                }
            except (pickle.PicklingError, TypeError):
                # print('Could not pickle object:', type(obj))
                out_data = None

            return out_data

        # Serialize using class name if no custom function
        # obj_dict = {
        #     '__type__': obj.__class__.__name__,
        #     'attributes': {}
        # }
        # for key, value in vars(obj).items():
        #     obj_dict['attributes'][key] = serialize(value)
        # return obj_dict

def deserialize(data):
    """Recursively deserialize a JSON-compatible structure."""

    if isinstance(data, dict):
        if '__type__' in data:
            class_name = data['__type__']

            if class_name == 'NumpyArray':
                return NumpyArray_Deserialize(data)
            
            # Special handling for pickled data
            elif class_name == 'PickleSerialized':
                return pickle.loads(bytes.fromhex(data['data']))

            # Look up the class
            obj_class = globals().get(class_name)


            # Check for custom deserialization
            deserialize_func_name = f"{class_name}_Deserialize"
            if deserialize_func := getattr(obj_class, deserialize_func_name, None):
                return deserialize_func(data)
            elif deserialize_func := globals().get(deserialize_func_name, None):
                return deserialize_func(data)
            else:
                raise ValueError(f"Deserialization function '{deserialize_func_name}' not found.")
            
            # Reconstruct object if it has attributes
            # obj = obj_class.__new__(obj_class)
            # for key, value in data['attributes'].items():
            #     setattr(obj, key, deserialize(value))
            # return obj
        else:
            return {key: deserialize(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [deserialize(item) for item in data]
    elif is_json_compatible(data):
        return data
    else:
        raise TypeError("Cannot deserialize object")



def NumpyArray_Serialize(obj: np.ndarray):
    # return obj.tolist()
    return {'__type__': 'NumpyArray', 'data': obj.tolist()}

def NumpyArray_Deserialize(data):
    return np.array(data['data'])

def tuple_Serialize(obj: tuple):
    return {'__type__': 'tuple', 'data': obj}

def tuple_Deserialize(data):
    return tuple(data['data'])

def SignFlipper_Serialize(self: SignFlipper) -> dict:
    return {
        'hist_length': self.hist_length,
        'n_states': self.n_states,
        'n_nuc': self.n_nuc,
        'nac_hist': self.nac_hist,
        'tdm_hist': self.tdm_hist,
        'name': self.name
    }

def SignFlipper_Deserialize(data: dict) -> 'SignFlipper':
    from .interpolation import SignFlipper
    sf = SignFlipper(data['n_states'], data['hist_length'], data['n_nuc'], data['name'])
    sf.nac_hist = data['nac_hist']
    sf.tdm_hist = data['tdm_hist']
    return sf

def TCRunner_Serialize(self: 'TCRunner') -> dict:
    '''
        as of now, this is used for a restart file. TODO: make it more general
    '''
    from .qcRunners import TCRunner
    out_data = {}
    for key in ['_atoms', '_grads', '_max_state', '_NACs', '_excited_type']:
        out_data[key] = serialize(getattr(self, key))
    
    out_data['_prev_results'] = TCRunner.cleanup_multiple_jobs(self._prev_results)
    out_data['_prev_job_batch'] = None
    

    for client in self._client_list:
        if client._exciton_data is not None and out_data.get('overlap_data', None) is None:
            out_data['overlap_data'] = serialize(client._exciton_data)

        if client._exciton_overlap_data is not None and out_data.get('exciton_overlap_data', None) is None:
            exciton_overlap_data = client._exciton_overlap_data
            out_data['exciton_overlap_data'] = base64.b64encode(pickle.dumps(exciton_overlap_data)).decode('utf-8')
        
        if client._scf_guess_file is not None and out_data.get('scf_guess', None) is None:
            with open(client._scf_guess_file, 'rb') as file:
                out_data['scf_guess'] = base64.b64encode(file.read()).decode('utf-8')
        
        if client._cis_guess_file is not None and out_data.get('cis_guess', None) is None:
            cis_file = client.server_file(client._cis_guess_file)
            with open(cis_file, 'rb') as file:
                out_data['cis_guess'] = base64.b64encode(file.read()).decode('utf-8')
        
        if client._cas_guess_file is not None and out_data.get('cas_guess', None) is None:
            with open(client._cas_guess_file, 'rb') as file:
                out_data['cas_guess'] = base64.b64encode(file.read()).decode('utf-8')

    return out_data

def TCRunner_Deserialize(data: dict, tc_runner: 'TCRunner'):
    '''
        as of now, this is used for a restart file. TODO: make it more general
    '''
    print('IN TCRUNNER DESERIALIZE')
    key_descriptions = {
        '_atoms': 'atoms',
        '_grads': 'gradients',
        '_max_state': 'maximum state',
        '_NACs': 'non-adiabatic couplings',
        '_excited_type': 'excited state type'
    }
    for k, d in key_descriptions.items():
        if k not in data:
            print(f'Warning: {d} not found in restart file')
        runner_val = getattr(tc_runner, k)
        data_val = deserialize(data.get(k, None))
        if runner_val != data_val:
            print(f'Warning: {d} in restart file does not match the current value')
            if k in ['_max_state', '_excited_type']:
                print(f'Current value: {runner_val}, Restart value: {data_val}')
            else:
                print('Current    Restart')
                for v1, v2 in zip_longest(runner_val, data_val):
                    print(f'{v1}    {v2}')

    if '_prev_results' in data:
        tc_runner._prev_results = data['_prev_results']

    if 'exciton_overlap_data' in data:
        overlap_data = pickle.loads(base64.b64decode(data['exciton_overlap_data']))
        tc_runner._coordinate_exciton_overlap_files(overlap_data=overlap_data)

    for client in tc_runner._client_list:
        if 'scf_guess' in data:
            tmp_scf_guess_file = client.server_file('scf_guess')
            with open(tmp_scf_guess_file, 'wb') as file:
                file.write(base64.b64decode(data['scf_guess']))
            client._scf_guess_file = tmp_scf_guess_file

        if 'cis_guess' in data:
            cis_guess_file = client.server_file(tc_runner._excited_options['cisrestart'])
            with open(cis_guess_file, 'wb') as file:
                file.write(base64.b64decode(data['cis_guess']))
            client._cis_guess_file = cis_guess_file

        if 'cas_guess' in data:
            tmp_cas_guess_file = client.server_file('cas_guess')
            with open(tmp_cas_guess_file, 'wb') as file:
                file.write(base64.b64decode(data['cas_guess']))
            client._cas_guess_file = tmp_cas_guess_file

    # exit()

        


# Example usage
if __name__ == '__main__':
    # Example data for serialization and deserialization
    array = np.array([1, 2, 3])
    serialized_data = serialize(array)
    json_data = json.dumps(serialized_data, indent=4)

    # Deserialize
    deserialized_data = deserialize(json.loads(json_data))
    print("Original:", array)
    print("Serialized JSON:", json_data)
    print("Deserialized:", deserialized_data)

