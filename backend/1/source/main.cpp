#include "image.h"
#include "iostream"
#include "cstring"
#include "string"
#include <complex>
using namespace std;

/*
g++ -o main main.cpp image.cpp -std=c++17
*/

int main(int argc, char *argv[])
{
  for (int i = 0; i < argc; i++) // Consider removing the loop.
  {
    if (std::strcmp(argv[1], "graylum") == 0)
    {
      Image test(argv[3]);

      Image gray_lum = test;
      gray_lum.grayscale_lum();
      gray_lum.write(argv[2]); // Could also convert using std::string(argv[2]).c_str()
      return 0;
    }
    else if (std::strcmp(argv[1], "copy") == 0)
    {
      Image test(argv[2]);
      // test.write("new.png");

      Image copy = test;

      for (int y = 0; y < copy.h / 2; ++y)
      {
        for (int x = 0; x < copy.w; ++x)
        {
          int pixelIndex = (y * copy.w + x) * copy.channels;
          for (int c = 0; c < copy.channels; ++c)
          {
            copy.data[pixelIndex + c] = 0;
          }
        }
      }
      copy.write(argv[3]);

      // Image blank(100, 100, 3);
      // blank.write("blank.jpg");
    }
    else if (std::strcmp(argv[i], "noisefilter") == 0)
    {
      Image img(argv[2]);

      if (!img.read(argv[2]))
      {
        std::cerr << "Failed to load input_image.jpg" << std::endl;
        return -1;
      }
      // Apply the noise filtering and get a new filtered image
      Image filteredImg = img.applyNoiseFiltering();

      // Save the filtered image as a new JPEG
      if (!filteredImg.write(argv[3]))
      {
        std::cerr << "Failed to save filtered_image.jpg" << std::endl;
        return -1;
      }
      std::cout << "Filtered image saved as filtered_image.jpg" << std::endl;
      return 0;
    }
    else if (std::strcmp(argv[i], "colormask") == 0)
    {
      Image img(argv[2]);

      if (!img.read(argv[2]))
      {
        std::cerr << "Failed to load input_image.jpg" << std::endl;
        return -1;
      }

      // Default RGB values (0, 0, 1)
      int r = 0, g = 1, b = 0;  

      // If user provides enough arguments,
      if (argc > 4) r = std::atoi(argv[4]);
      if (argc > 5) g = std::atoi(argv[5]);
      if (argc > 6) b = std::atoi(argv[6]);

      img.colorMask(r, g, b);
      img.write(argv[3]);
      // printf("Got here!");
      return 0;
    } else if (std::strcmp(argv[i], "crop") == 0) {
      printf("%s, %s", argv[0], argv[1]);
      Image img(argv[2]);
      img.write("temp.png");
      img.crop(0, 0, 300, 300);
      img.write(argv[3]);
    } else if (std::strcmp(argv[i], "diffmap") == 0)
    {
      Image img("cat_test.jpg");
      Image img2("dog_test.jpg");
      Image diff = img;
      diff.diffmap(img2);
      diff.write("diff.png");
    }
    else if (std::strcmp(argv[i], "flipx") == 0)
    {
      Image img("test.jpg");
      img.flipX();
      img.write("flipped.png");
    }
    else if (std::strcmp(argv[i], "flipy") == 0)
    {
      Image img("cat_test.jpg");
      img.flipY();
      img.write("flippedy.png");
    }
    else if (std::strcmp(argv[i], "edge_detection") == 0)
    {
      Image img(argv[2]);

	
      // grayscale
      img.grayscale_avg();
      int img_size = img.w*img.h;
    
      Image gray_img(img.w, img.h, 1);
      for(uint64_t k=0; k<img_size; ++k) {
        gray_img.data[k] = img.data[img.channels*k];
      }
      gray_img.write("imgs/test6_gray.png");
    
      
      // blur
      Image blur_img(img.w, img.h, 1);
      double gauss[9] = {
        1.0/16.0, 2.0/16.0, 1.0/16.0,
        2.0/16.0, 4.0/16.0, 2.0/16.0,
        1.0/16.0, 2.0/16.0, 1.0/16.0
    };
      // double gauss[9] = {
      //   1/16., 2/16., 1/16.,
      //   2/16., 4/16., 2/16.,
      //   1/16., 2/16., 1/16.
      // };
      gray_img.convolve_linear(0, 3, 3, gauss, 1, 1);
      for(uint64_t k=0; k<img_size; ++k) {
        blur_img.data[k] = gray_img.data[k];
      }
      blur_img.write("imgs/test6_blur.png");
    
    
      // edge detection
      double* tx = new double[img_size];
      double* ty = new double[img_size];
      double* gx = new double[img_size];
      double* gy = new double[img_size];
    
      //seperable convolution
      for(uint32_t c=1; c<blur_img.w-1; ++c) {
        for(uint32_t r=0; r<blur_img.h; ++r) {
          tx[r*blur_img.w+c] = blur_img.data[r*blur_img.w+c+1] - blur_img.data[r*blur_img.w+c-1];
          ty[r*blur_img.w+c] = 47*blur_img.data[r*blur_img.w+c+1] + 162*blur_img.data[r*blur_img.w+c] + 47*blur_img.data[r*blur_img.w+c-1];
        }
      }
      for(uint32_t c=1; c<blur_img.w-1; ++c) {
        for(uint32_t r=1; r<blur_img.h-1; ++r) {
          gx[r*blur_img.w+c] = 47*tx[(r+1)*blur_img.w+c] + 162*tx[r*blur_img.w+c] + 47*tx[(r-1)*blur_img.w+c];
          gy[r*blur_img.w+c] = ty[(r+1)*blur_img.w+c] - ty[(r-1)*blur_img.w+c];
        }
      }
    
      delete[] tx;
      delete[] ty;
    
      //make test images
      double mxx = -INFINITY,
        mxy = -INFINITY,
        mnx = INFINITY,
        mny = INFINITY;
      for(uint64_t k=0; k<img_size; ++k) {
        mxx = fmax(mxx, gx[k]);
        mxy = fmax(mxy, gy[k]);
        mnx = fmin(mnx, gx[k]);
        mny = fmin(mny, gy[k]);
      }
      Image Gx(img.w, img.h, 1);
      Image Gy(img.w, img.h, 1);
      for(uint64_t k=0; k<img_size; ++k) {
        Gx.data[k] = (uint8_t)(255*(gx[k]-mnx)/(mxx-mnx));
        Gy.data[k] = (uint8_t)(255*(gy[k]-mny)/(mxy-mny));
      }
      Gx.write("imgs/Gx.png");
      Gy.write("imgs/Gy.png");
    
    
      // fun part
      double threshold = 0.09;
      double* g = new double[img_size];
      double* theta = new double[img_size];
      double x, y;
      for(uint64_t k=0; k<img_size; ++k) {
        x = gx[k];
        y = gy[k];
        g[k] = sqrt(x*x + y*y);
        theta[k] = atan2(y, x);
      }
    
      //make images
      double mx = -INFINITY,
        mn = INFINITY;
      for(uint64_t k=0; k<img_size; ++k) {
        mx = fmax(mx, g[k]);
        mn = fmin(mn, g[k]);
      }
      Image G(img.w, img.h, 1);
      Image GT(img.w, img.h, 3);
    
      double h, s, l;
      double v;
      for(uint64_t k=0; k<img_size; ++k) {
        //theta to determine hue
        h = theta[k]*180./M_PI + 180.;
    
        //v is the relative edge strength
        if(mx == mn) {
          v = 0;
        }
        else {
          v = (g[k]-mn)/(mx-mn) > threshold ? (g[k]-mn)/(mx-mn) : 0;
        }
        s = l = v;
    
        //hsl => rgb
        double c = (1-abs(2*l-1))*s;
        double x = c*(1-abs(fmod((h/60),2)-1));
        double m = l-c/2;
    
        double rt, gt, bt;
        rt=bt=gt = 0;
        if(h < 60) {
          rt = c;
          gt = x;
        }
        else if(h < 120) {
          rt = x;
          gt = c;
        }
        else if(h < 180) {
          gt = c;
          bt = x;
        }
        else if(h < 240) {
          gt = x;
          bt = c;
        }
        else if(h < 300) {
          bt = c;
          rt = x;
        }
        else {
          bt = x;
          rt = c;
        }
    
        uint8_t red, green, blue;
        red = (uint8_t)(255*(rt+m));
        green = (uint8_t)(255*(gt+m));
        blue = (uint8_t)(255*(bt+m));
    
        GT.data[k*3] = red;
        GT.data[k*3+1] = green;
        GT.data[k*3+2] = blue;
    
        G.data[k] = (uint8_t)(255*v);
      }
      G.write("imgs/edge_detected.png");
      GT.write("imgs/edge_detected_color.png");
    
      delete[] gx;
      delete[] gy;
      delete[] g;
      delete[] theta;
    
      
      return 0;
    // std::cout << "Index: ";
    // std::cout << i << std::endl;
    // std::cout << argv[i] << std::endl;
  }
}
}


// int main(int argc, char** argv) {
// Image test("test1.jpg");

// Image gray_lum = test;
// gray_lum.grayscale_lum();
// gray_lum.write("gray_lum.png");
// return 0;

//   test.write("new.png");

//   Image copy = test;
//   // for (int j = 0; j < (copy.h); ++j) {
//   //   for (int i = 0; i < copy.w*copy.channels; ++i) {
//   //     copy.data[i] = 255;
//   //   };
//   // }

//   for (int y = 0; y < copy.h / 2; ++y) {
//     for (int x = 0; x < copy.w; ++x) {
//         int pixelIndex = (y * copy.w + x) * copy.channels;
//         for (int c = 0; c < copy.channels; ++c) {
//             copy.data[pixelIndex + c] = 0;
//         }
//     }
// }
//   copy.write("copy.png");

//   Image blank(100, 100, 3);
//   blank.write("blank.jpg");

// }