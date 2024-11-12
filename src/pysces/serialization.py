import json
import numpy as np
import importlib
import inspect
from types import ModuleType
import pickle

from .interpolation import SignFlipper

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
    """Recursively serialize an object."""

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
    else:

        # Use custom serialization function if available on the class
        serialize_func_name = f"{obj.__class__.__name__}_Serialize"
        if serialize_func := getattr(obj.__class__, serialize_func_name, None):
            return serialize_func(obj)
        
        # Use pickle for objects without a __dict__ attribute
        if not hasattr(obj, '__dict__'):
            try:
                # pickle.dumps(obj).hex()
                out_data = {
                    '__type__': 'PickleSerialized',
                    'data': pickle.dumps(obj).hex()
                }
                print('Pickle serialized object:', type(obj))
            except (pickle.PicklingError, TypeError):
                print('Could not pickle object:', type(obj))
                out_data = None

            return out_data

        # Serialize using class name if no custom function
        obj_dict = {
            '__type__': obj.__class__.__name__,
            'attributes': {}
        }
        for key, value in vars(obj).items():
            obj_dict['attributes'][key] = serialize(value)
        return obj_dict

def deserialize(data):
    """Recursively deserialize a JSON-compatible structure."""
    if is_json_compatible(data):
        return data
    elif isinstance(data, list):
        return [deserialize(item) for item in data]
    elif isinstance(data, dict):
        if '__type__' in data:
            class_name = data['__type__']
            
            # Special handling for pickled data
            if class_name == 'PickleSerialized':
                return pickle.loads(bytes.fromhex(data['data']))

            # Look up the class
            obj_class = globals().get(class_name)
            if not obj_class:
                raise ValueError(f"Class '{class_name}' not found in globals.")

            # Check for custom deserialization
            deserialize_func_name = f"{class_name}_Deserialize"
            if deserialize_func := getattr(obj_class, deserialize_func_name, None):
                return deserialize_func(data)
            
            # Reconstruct object if it has attributes
            obj = obj_class.__new__(obj_class)
            for key, value in data['attributes'].items():
                setattr(obj, key, deserialize(value))
            return obj
        elif '__type__' in data and data['__type__'] == 'NumpyArray':
            return NumpyArray_Deserialize(data)
        else:
            return {key: deserialize(value) for key, value in data.items()}
    else:
        raise TypeError("Cannot deserialize object")



def NumpyArray_Serialize(obj):
    return {
        '__type__': 'NumpyArray',
        'data': obj.tolist()
    }

def NumpyArray_Deserialize(data):
    return np.array(data['data'])

def SignFlipper_serialize(self) -> dict:
    return {
        'hist_length': self.hist_length,
        'n_states': self.n_states,
        'n_nuc': self.n_nuc,
        'nac_hist': self.nac_hist,
        'tdm_hist': self.tdm_hist,
        'name': self.name
    }

def SignFlipper_deserialize(data: dict) -> 'SignFlipper':
    sf = SignFlipper(data['n_states'], data['hist_length'], data['n_nuc'], data['name'])
    sf.nac_hist = data['nac_hist']
    sf.tdm_hist = data['tdm_hist']
    return sf

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

