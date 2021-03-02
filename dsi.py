import os
import tensorflow as tf
import numpy as np

from utility import _create_patient_data_df, _get_file_names, _parse_dicom_data

class OSICFibrosisDSI:
    def __init__(self, csv_path, ct_path):
        self.csv_path = csv_path
        self.ct_path = ct_path

    def _load_data(self):
        self._test_patient_data_df = _create_patient_data_df(self.csv_path)
        
        all_files = _get_file_names(self.ct_path)
        dataset = tf.data.Dataset.from_tensor_slices(all_files)
        num_data = len(all_files)
        
        dataset = dataset.map(_parse_dicom_data)
        dataset = dataset.map(lambda x: self._add_patient_data_map(x))

        return dataset, num_data

    def get_test_dataset(self):
        batch_size = 1
        dataset, num_data = self._load_data()
        dataset = dataset.batch(batch_size)
        
        # use 1 for test batch size
        return dataset, num_data, batch_size
    
    def _get_patient_data(self, patient_id):
        """ Gets wrapped in tf.numpy_func, gets patient data
        """
        patient_id = patient_id.decode("utf-8")

        patient_df = self._test_patient_data_df
        data = patient_df[
            patient_df.Patient == patient_id
            ].iloc[0][1:].values.tolist()
        
        # Ensure all elements are arrays so they can be concatenated
        for i in range(len(data)):
            if isinstance(data[i], np.ndarray):
                pass
            else:
                data[i] = np.array([data[i]])
        
        data = np.concatenate(data).astype(np.float32)
        return data
    
    def _add_patient_data_map(self, data_tsr):
        """ Adds patient metadata to tsr containing just patient_id and image
        """
        data_vec = tf.numpy_function(
            self._get_patient_data, [data_tsr['Patient']], Tout=tf.float32)
        data_vec = tf.split(data_vec, [4,1,1])
        
        data_tsr['metadata'] = tf.ensure_shape(data_vec[0], [4])
        data_tsr['cur_fvc'] = tf.ensure_shape(data_vec[1], [1])
        data_tsr['cur_week'] = tf.ensure_shape(data_vec[2], [1])

        return data_tsr