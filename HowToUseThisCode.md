# How to start the code
Putting aside the facts that you have Python, the right modules and opencv 3 installed on your computer and that you have checked all the dependencies or course. Then you should be able to run the bazard.

## Getting some help
Beforet to start a live session of the image pixel mirror you can check for info like this:

```
>python startPixelMirror.py -h
We process the input arguments.
usage: startPixelMirror.py [-h] [-dbs DATABASE_SIZE] -nr NUMBER_ROWS -dbp
                           DATABASE_PATH [-nf NUMBER_FRAMES]
                           [-nfu NUMBER_FRAMES_FOR_UPDATE]

optional arguments:
  -h, --help            show this help message and exit
  -dbs DATABASE_SIZE, --database_size DATABASE_SIZE
                        number of images in the database
  -nr NUMBER_ROWS, --number_rows NUMBER_ROWS
                        number of cell/super-pixel for the final mosaic, eg 4
                        x 4 if nr = 4
  -dbp DATABASE_PATH, --database_path DATABASE_PATH
                        path to where to store or read the database of images
  -nf NUMBER_FRAMES, --number_frames NUMBER_FRAMES
                        the program stops after 1000 frames or the specidied
                        amount given
  -nfu NUMBER_FRAMES_FOR_UPDATE, --number_frames_for_update NUMBER_FRAMES_FOR_UPDATE
                        the program stops after 1000 frames or the specidied
                        amount given
```

## Starting the bazard
Let's say you want to create a mosaic of 8x8 cell having the same ratio of the orignal frame, update the database of images used as pixel mosaix of 128 images every 20 frames you will do:

```
>python startPixelMirror.py -dbs 128 -nr 8 -dbp frame_db -nfu 20
```

## And after?
When you press q it should stop. 

When you start again with the same settings it should delete the previous database if you choose the same name.