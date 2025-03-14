# MEDICAL IMAGE PROCESSING


## USAGE
Navigate to the home directory. The functions required is presently located in the image.cpp and main.cpp file so they are both required during compilation. The code can be compiled using the g++ compiler:

```g++ -o main main.cpp image.cpp -std=c++17```

The code presently supports the following arguments.

```
> -  graylum
> -  copy
> -  noisefilter
> -  crop
> -  colormask
```

* Graylum: Creates a grayscale version of the image. 
* Copy: Creates a copy version of the image. 
* noisefilter: Creates a filtered version of the image. 
* Crop: Crops a particular portion of the image out. It could be used to crop specific parts of the image out for analysis.
* Colormask: Colormask helps define the rgb levels of image within the entire image.

The program can be run from the terminal using the following format:

From the compile statement written above, the compiled code would be located in the file called "main." main gets outputted as an executable file with permissions rwxr-xr-x (755). 

`- "./main" should always be the first command on the terminal and then the following args follow. 


Here is a list of how arguments should be ordered when being run. For functions that require an alternative arrangement, they are listed further below.

* First arg: Name of functionality to be run. e.g "graylum", "noisefilter", "copy", "crop"
* Second arg: Name of Image to be processed. e.g 'test.jpg'or 'test.png'
* Third arg: Name of the output image (Preferably .png or .jpg). E.g 'output.png'



# Example Usage
```./main graylum test.jpg output.png```

This creates a graylum version of the image.

Colormask requires extra arguments for its dynamic rgb values. To input that, after the regular argument, input the rgb values after. Example:

* > ./main colormask test.jpg colormasked.png r g b
* > ./main colormask test.jpg colormasked.png 120 120 120

if user doesn't provide values for the rgb parameter, it defaults to (0, 1, 0) which defaults to green.