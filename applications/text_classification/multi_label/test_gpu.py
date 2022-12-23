import tensorflow as tf
if __name__ == '__main__':
    # 查看gpu和cpu的数量
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
    print(gpus, cpus)
