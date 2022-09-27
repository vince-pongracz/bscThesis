# BscThesis

### Téma:
Angol témacím: Applied artifical intelligence on Java, Node.JS or on other platform

Témacím: Alkalmazott Mesterséges Intelligencia Java, Node.JS vagy egyéb platformon

English title of the thesis: Examination and application of image processing algorithms

Szakdolgozat címe: Képfeldolgozási algoritmusok vizsgálata és alkalmazása

Konzulens: Dr. Ekler Péter

---

[Link to Drive](https://drive.google.com/drive/folders/1F7TlPfy6_YdyRuGlejn6DzYuy2ElJa39?usp=sharing)

### Ideas:
 - face/object recognition
 - semantic segmentation
 - share/stock market analysis
 - data stream handling (?)
 - __images --> learn good exposure/retouch values on image (exposure, contrast, white balance values), apply this to other images -- automatic retouch (hard to get train this?)__
 - fake profile generator (generate personal data, like picture, name, place of birth, time of birth, etc...)
 - exposure prediction (shutter, ISO, aperture values) - many many raw pictures without retouch required?
 - classify pictures (sharp/obscure, moved in)
 - detect joint-crops on images

### Choosen: 
 - images --> learn good exposure/retouch values on image (exposure, contrast, white balance values), apply this to other images -- automatic retouch (hard to get train this?)

### Tasks, subtasks
 - jpg processing:
    - use small jpgs from spotweb (jpeg recompress)
    - transform/reshape jpgs to the same size
 - process jpgs with CNN (get train, test, valid sets)
    - #ofHiddenLayers ?
    - input: jpg - uniform size (only landscape pictures?)
    - output: json like this:
    ```json
        {
            "exposure": -0.23,
            "colorTemperature": 6750,
            "tint": -5,
            "contrast": 7
        }
    ```
    - error function: 
        - multiple error funtion for the different features?
        - multiple parallel CNN to learn the different features?

        MSE based on predicted and actual exif fields - compare the json with the fields of the exifdata
    - optimizer: Adam is OK and effective enough
 - train the NN, fix the trained NN
    - dropout
    - in the beginning -- small dataset, but many epochs
 - Preprocess raw pictures
    - resize/reshape raw images passing to the NN input dim. (crop at the edges?)
 - NN returns the json as output
 - generate .xmp file to raw image (same name as the raw)
 - rewrite fields of originally created .xmp from raw -- based on the json
    - process .xmp (load & write)
 - load automate generated .xmp-s and raws into Lightroom or other image processing software
 - raw + .xmp = jpg processed - do the real job, create the retouched image

### Additonal:
 - batch processing - read and make prediction on multiple raws
 - develop an easy GUI:
     - load pretrained NN model
     - train an NN based on dir of jpgs
     - generate&predict .xmp for a raw file
     - generate&predict .xmp for dir of raw files

### Language & Tech:
 - python seems ok for the task...
 - desktop application