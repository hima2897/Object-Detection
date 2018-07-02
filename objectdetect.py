
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import glob
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
import io
import tensorflow as tf
import errno
import PIL
import json
import functools

from PIL import Image
from utils import dataset_util
from collections import namedtuple, OrderedDict

from object_detection import trainer
from builders import dataset_builder
from builders import graph_rewriter_builder
from builders import model_builder
from utils import config_util
from utils import dataset_util

tf.logging.set_verbosity(tf.logging.INFO)

image_path = 'annotations'


flags = tf.app.flags
# flags.DEFINE_string('image_path', '', 'Path to the images folder')
# flags.DEFINE_string('XML_path', '', 'Path to')
flags.DEFINE_string('master', '', 'Name of the TensorFlow master to use.')
flags.DEFINE_integer('task', 0, 'task id')
flags.DEFINE_integer('num_clones', 1, 'Number of clones to deploy per worker.')
flags.DEFINE_boolean('clone_on_cpu', False,
                     'Force clones to be deployed on CPU.  Note that even if '
                     'set to False (allowing ops to run on gpu), some ops may '
                     'still be run on the CPU if they have no GPU kernel.')
flags.DEFINE_integer('worker_replicas', 1, 'Number of worker+trainer '
                     'replicas.')
flags.DEFINE_integer('ps_tasks', 0,
                     'Number of parameter server tasks. If None, does not use '
                     'a parameter server.')
flags.DEFINE_string('train_dir', '',
                    'Directory to save the checkpoints and training summaries.')

flags.DEFINE_string('pipeline_config_path', '',
                    'Path to a pipeline_pb2.TrainEvalPipelineConfig config '
                    'file. If provided, other configs are ignored')

flags.DEFINE_string('train_config_path', '',
                    'Path to a train_pb2.TrainConfig config file.')
flags.DEFINE_string('input_config_path', '',
                    'Path to an input_reader_pb2.InputReader config file.')
flags.DEFINE_string('model_config_path', '',
                    'Path to a model_pb2.DetectionModel config file.')

FLAGS = flags.FLAGS


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    print('Successfully converted xml to csv.')
    return xml_df


def tfRecord(csv_input, image_path, output_path):
    # TO-DO replace this with label map
    def class_text_to_int(row_label):
        pet_label_map ="item {\n id : 1\n name :"+" '"+row_label +"'\n }"
        with open('pet_label_map.pbtxt','w') as txt:
            txt.write(pet_label_map)
        return 1


    def split(df, group):
        data = namedtuple('data', ['filename', 'object'])
        gb = df.groupby(group)
        return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


    def create_tf_example(group, path):
        with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
            encoded_jpg = fid.read()
        encoded_jpg_io = io.BytesIO(encoded_jpg)
        image = Image.open(encoded_jpg_io)
        width, height = image.size

        filename = group.filename.encode('utf8')
        image_format = b'jpg'
        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        for index, row in group.object.iterrows():
            xmins.append(row['xmin'] / width)
            xmaxs.append(row['xmax'] / width)
            ymins.append(row['ymin'] / height)
            ymaxs.append(row['ymax'] / height)
            classes_text.append(row['class'].encode('utf8'))
            classes.append(class_text_to_int(row['class']))

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': dataset_util.int64_feature(height),
            'image/width': dataset_util.int64_feature(width),
            'image/filename': dataset_util.bytes_feature(filename),
            'image/source_id': dataset_util.bytes_feature(filename),
            'image/encoded': dataset_util.bytes_feature(encoded_jpg),
            'image/format': dataset_util.bytes_feature(image_format),
            'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
            'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
            'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
            'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
            'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
            'image/object/class/label': dataset_util.int64_list_feature(classes),
        }))
        return tf_example
    writer = tf.python_io.TFRecordWriter(output_path)
    path = 'images'
    examples = pd.read_csv(csv_input)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, path)
        writer.write(tf_example.SerializeToString(output_path))

    writer.close()
    output_path = os.path.join(os.getcwd(), output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))

def main(_):
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('raccoon.csv', index=None)
    df = pd.read_csv('raccoon.csv')
    df['split'] = np.random.randn(df.shape[0], 1)

    msk = np.random.rand(len(df)) <= 0.75

    train = df[msk]
    test = df[~msk]
    train.to_csv('train_labels.csv', index=False)
    test.to_csv('test_labels.csv', index=False)
    print('Successfully split into train and test csv files.')
    # train_record = 'train.record'
    # test_record = 'test.record'
    tfRecord('train_labels.csv', image_path, 'train.record')
    tfRecord('test_labels.csv',image_path,'test.record')
    
    train_dir = 'data'
    pipeline_config_path = 'ssd_mobilenet_v1_pets.config'
    if FLAGS.task == 0: tf.gfile.MakeDirs(train_dir)
    if pipeline_config_path:
        configs = config_util.get_configs_from_pipeline_file(pipeline_config_path)
        if FLAGS.task == 0:
            tf.gfile.Copy(pipeline_config_path,
                    os.path.join(train_dir, 'pipeline.config'),
                    overwrite=True)
    else:
        configs = config_util.get_configs_from_multiple_files(
                model_config_path=FLAGS.model_config_path,
                train_config_path=FLAGS.train_config_path,
                train_input_config_path=FLAGS.input_config_path)
        if FLAGS.task == 0:
            for name, config in [('model.config', FLAGS.model_config_path),
                            ('train.config', FLAGS.train_config_path),
                            ('input.config', FLAGS.input_config_path)]:
                tf.gfile.Copy(config, os.path.join(train_dir, name),overwrite=True)
    
    model_config = configs['model']
    train_config = configs['train_config']
    input_config = configs['train_input_config']

    model_fn = functools.partial(
        model_builder.build,
        model_config=model_config,
        is_training=True)

    def get_next(config):
        return dataset_util.make_initializable_iterator(
            dataset_builder.build(config)).get_next()

    create_input_dict_fn = functools.partial(get_next, input_config)

    env = json.loads(os.environ.get('TF_CONFIG', '{}'))
    cluster_data = env.get('cluster', None)
    cluster = tf.train.ClusterSpec(cluster_data) if cluster_data else None
    task_data = env.get('task', None) or {'type': 'master', 'index': 0}
    task_info = type('TaskSpec', (object,), task_data)

    # Parameters for a single worker.
    ps_tasks = 0
    worker_replicas = 1
    worker_job_name = 'lonely_worker'
    task = 0
    is_chief = True
    master = ''

    if cluster_data and 'worker' in cluster_data:
        # Number of total worker replicas include "worker"s and the "master".
        worker_replicas = len(cluster_data['worker']) + 1
    if cluster_data and 'ps' in cluster_data:
        ps_tasks = len(cluster_data['ps'])

    if worker_replicas > 1 and ps_tasks < 1:
        raise ValueError('At least 1 ps task is needed for distributed training.')

    if worker_replicas >= 1 and ps_tasks > 0:
        # Set up distributed training.
        server = tf.train.Server(tf.train.ClusterSpec(cluster), protocol='grpc',
                                job_name=task_info.type,
                                task_index=task_info.index)
        if task_info.type == 'ps':
            server.join()
            return

        worker_job_name = '%s/task:%d' % (task_info.type, task_info.index)
        task = task_info.index
        is_chief = (task_info.type == 'master')
        master = server.target

    graph_rewriter_fn = None
    if 'graph_rewriter_config' in configs:
        graph_rewriter_fn = graph_rewriter_builder.build(
            configs['graph_rewriter_config'], is_training=True)

    trainer.train(
        create_input_dict_fn,
        model_fn,
        train_config,
        master,
        task,
        FLAGS.num_clones,
        worker_replicas,
        FLAGS.clone_on_cpu,
        ps_tasks,
        worker_job_name,
        is_chief,
        train_dir,
        graph_hook_fn=graph_rewriter_fn)

if __name__ == '__main__':
    tf.app.run()
