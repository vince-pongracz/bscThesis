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
 - desktop application:
    - python + GUI
    - C# + WPF???
- node?

### Dolgozat felépítése, összefoglaló

Dolgozat felépítés példa vázlat:

#### Összefoglaló (1 oldal)
#### Abstract (1 oldal)
#### Bevezetés (2-3 oldal)
  - Témaválasztás indoklása
  - Felhasznált technológia jelentősége/elterjedtsége
  - Utolsó bekezdésben: melyik fejezet miről fog szólni
#### Feladatspecifikáció (3-4 oldal)
  - Feladat részletes leírása
  - Use case diagram: program funkciói (lehetőleg bonyolultabb pl. entitások öröklődése)
  - Activity diagram
#### Irodalomkutatás (5 oldal)
 - Felhasznált technológiák
 - Hasonló megoldások (cikkek, könyvek, hasonló rendszerekről) http://scholar.google.com (google: facebook statistics )
#### Felsőszintű architektúra
 - High level architektúra ábra
 - Rendszer felépítései, komponensei
#### Részletes megvalósítás
 - UML class diagramok
 - Enity-relation diagram
 - Szekvencia diagram
 - Jó rajzolók: http://plantuml.com/, https://sequencediagram.org/, https://www.draw.io/
 - Kódrészek
#### Tesztelés: pl JMeterrel a szervert vagy Loader.io
 - Felhasználói leírás
 - Screenshotokkal elmagyarázni hogy kell használni a programot
#### Összefoglalás, továbbfejlesztési lehetőségek

Amire figyelni kell:

Szakdolgozat/Diploma sablon és fontos infok: https://www.aut.bme.hu/Pages/Gyik/Diploma

Fontos információk: https://www.aut.bme.hu/Pages/Gyik/Onlab

Szakdolgozat legalább 45-50 oldal legyen, maximum ~65; Diplomaterv legalább 65 oldal legyen, maximum ~110

Két fejezet/alfejezet között ne legyen üres rész, kérlek nézd megkésőbb is mindenhol! Legalább annyit írj oda, hogy ez a szakasz miről fog szólni.

Próbálj szakmain, professzionálisan fogalmazni, kerüld a "szerintem" jellegű kifejezéseket. Gyakran érdemes passzívba fogalmazni, pl: "A következőkben bemutatom a megoldást" helyett "A következőkben a megoldás architektúrája kerül bemutatásra".

Ábrák olyan méretűek legyenek, hogy nyomtatásban is jól nézzen ki.

Webes hivatkozáskor a hivatkozás listában a weboldal utolsó megtekintési dátuma is legyen feltüntetve.

Legvégén ne felejtsd frissíteni a tartalom és ábrajegyzéket, mert az oldalszámok elcsúszhattak!

Az osztályok, függvények, kulcs szavak, stb. dőlt betűvel legyenek! A függvények után legyen ’()’ jel! Később ezt nem jelölöm, kérlek az egész dolgozatban nézd át!

Legvégén mindenképp futtass egy teljes helyesírás ellenőrzést, az instant ellenőrzés sokszor nem mutat minden hibát.

Legvégén érdemes mással átolvastatni!

UML rajzolásra pl http://staruml.io/, http://plantuml.com/, https://sequencediagram.org/, https://www.draw.io/ vagy Visio