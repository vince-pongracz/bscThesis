# BscThesis

### Témakiírás:
Angol témacím: Applied artifical intelligence on Java, Node.JS or on other platform

Témacím: Alkalmazott Mesterséges Intelligencia Java, Node.JS vagy egyéb platformon

### Dolgozat:
English title of the thesis: Examination and application of image processing algorithms (with neural nets)

Szakdolgozat címe: Képfeldolgozási algoritmusok vizsgálata és alkalmazása (neurális hálókkal)

Konzulens: Dr. Ekler Péter

---

[Link to organisational Drive](https://drive.google.com/drive/folders/1F7TlPfy6_YdyRuGlejn6DzYuy2ElJa39?usp=sharing)

[Link to Colab](https://colab.research.google.com/drive/1-etIrA0LBeHmrVwPMbkTgAu_wuZrVUUw?usp=sharing)

[Link to Colab-backing Drive](https://drive.google.com/drive/folders/1C2ijTf8recXGWYMgSxglKWcPdbKHO5_W?usp=share_link)

Some of these links maybe asks for special permission to access. Please contact me, if you meet some problems here.

### Idea:
__image processing: learn good exposure and some other retouch values on image (exposure, contrast, white balance, tint, vibrance values), and apply these to other images. So get an automatic retouch tool__

---

### Tasks, subtasks
 - raw processing:
 - process jpgs with CNN (get train, valid sets)
    - #ofHiddenLayers ?
    - input: raw converted to binary - uniform size
    - output: predicted values
    - error function:
        - multiple error funtion for the different features?
        - multiple parallel CNN to learn the different features?

        MSE based on predicted and actual exif fields - compare the output with the fields of the exifdata
    - optimizer: Adam is OK and effective enough, but let's try others
 - train the NNs, fix the trained NNs
    - dropout
    - in the beginning -- small dataset, but many epochs
 - NN returns output
 - generate .xmp file from output lists for raw image (same name as the raw)
 - rewrite fields of originally created .xmp from raw -- based on the lists
    - process .xmp (load & write)
 - load automate generated .xmp-s and raws into Lightroom or other image processing software