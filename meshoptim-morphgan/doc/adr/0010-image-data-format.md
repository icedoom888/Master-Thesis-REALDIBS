# 10. image data format

Date: 2020-03-25

## Status

In dicussion

## Context

The most essential component is the data storage. The core item are images which exist in
any file format known on earth and with different number of channels, e.g. RGBD or RGBA.


## Decision

The current decision how to encode the images files is.

primary data is stored as <filename>.<ext> as this is the raw input.
secondary data is stored as <filename>.png to losslessly store the image, with its
unique filename per folder only. the folder itself is stored with a <uuid> per asset.

note: the <uuid> is calculated on the md5 hash of the original file content + filename
in case there is the a collision, it is the exact same content anyway. e.g. skip.

the image content can be stored as RGB channel order and potentially following a 4th
channel, such as depth or alpha. This is in the secondary files.
For processing and learning only the RGB data is processed to be able to re-use
pretrained networks.

The conversion for RGB/BGR is to convert into standard RGB for all processes.
The conversion for RGBD is to skip the D channel and just keep RGB.
The conversion for RGBA is to copy the RGB data onto a white background.
Example is shown in the unit test: tests/test_image_format.py

The reason a white background is selected is because of human readability. Mostly the
images are shown on a brighter background and a black background (i.e. just copying RGB
and ignoring the 0s in the empty matrix), would make it difficult to check the images.

Hence, the tertiary data as TFDS records files reads in the <filename>.png, completes the
necessary conversion step, and stores the images as uint8 in the TF records. This is the
most space-saving form, and any later processing will be done during the TFDS dataset mapping
process. See dataset/base.py for any def _map(sample): function.

Reason: There is only one raw format, but many mappings needed. This way we only keep one
clean TFDS record and perform all mappings on-the-fly. E.g. normalization.

Finally, the most coherent system for writing TFDS records to fit in the current learning is:

multi-class: convert the single on class value to a one-hot vector e.g. [0 0 1 0 0] for 5 classes.
multi-label: also use a one-hot vector, unnormalized, e.g. [1 1 0 0 1 0] for 3 active classes.

additional fields may be specified in the TF records as per application need,
but note the following naming scheme to avoid issues:

```
    features=tfds.features.FeaturesDict(
        {
            "image": tfds.features.Image(shape=(w,h,c)),
            "labels": tfds.features.Tensor(
                shape=[None], dtype=tf.int64
            ),
            "labels_correct": tfds.features.Tensor(
                shape=None], dtype=tf.int64
            ),
        }
```

"image" is the image, "labels" is the one-hot vector used for training.
"labels_clean" is a specific label in case "labels" is corrupt to allow clean labels.

Additional names currently used are:

```
    features=tfds.features.FeaturesDict({
        "filename": tfds.features.Text(),
        "image": tfds.features.Image(),
        "labels": tfds.features.Tensor(shape=[None],dtype=tf.int64),
        "labelname": tfds.features.Text(),
    }),
```

"labelname" is a single string for this classes' textual label.
"filename" is the full path to the secondary data file source.

Best practices at the moment:
* image resolution is 224 x 224 x 3
* before creating labelset, sort alphabetically to maintain correct indices.
* int64 is just because...
* one-hot is comparately low memory even if label set cardinality is 1000s.


## Consequences

What becomes easier or more difficult to do and any risks introduced by the
change that will need to be mitigated.

We'll see how it goes. Schau ma mal, dann seh ma scho.
