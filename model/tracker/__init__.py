import tensorflow as tf

def get_embeddings(image, boxes, model, size):
    boxes_tensors = []
    for box in boxes:
        x1, y1, x2, y2 = box
        width = x2-x1
        height = y2-y1

        [bb_image] = tf.image.crop_to_bounding_box(
            image,
            int(y1*image.shape[1]),
            int(x1*image.shape[2]),
            int(height*image.shape[1]),
            int(width*image.shape[2])
        )
        bb_image = tf.image.resize(
            bb_image,
            size=(size, size)
        )
        boxes_tensors.append(bb_image)

    boxes_tensors = tf.stack(boxes_tensors, axis=0)

    return model.predict(boxes_tensors)
