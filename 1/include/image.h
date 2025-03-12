#include <iostream>
#include <stdint.h>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstddef>
#include <complex>

//It contains declaration of all functions the package is going to use.

// Setting up an image type.
enum ImageType {
  PNG, JPG, BMP, TGA
};

struct Image {
  uint8_t* data = NULL;
  size_t size = 0;
  int w;
  int h;
  int channels; // How many color values per pixel, rgb = 3, rgba = 4

  Image(const char* filename); // Readfile constructor.
  Image(int w, int h, int channels); // Constructor that creates a black image that we can manipulate willfully
  Image(const Image& img); // Copy constructor. Creates a copy of the image.
  ~Image();

  bool read(const char* filename); // Reads from a file (Called from the readfile constructor).
  bool write(const char* filename);

  ImageType getFileType(const char* filename);

  Image& grayscale_avg();
  Image& grayscale_lum();

  Image& colorMask(float r, float g, float b);

  // Cropping Images
  Image& crop(uint16_t cx, uint16_t cy, uint16_t cw, uint16_t ch);

  // Steganography
  Image& encodeMessage(const char* message); 
  Image& decodeMessage(char* buffer, size_t* messageLength); 

  Image& diffmap(Image& img);
  Image& diffmap_scale(Image& img, uint8_t scl = 0);

  // Flipping Image (Mirroring)
  Image& flipX();
  Image& flipY();

  	Image& convolve_linear(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);


  static void fft(uint32_t n, std::complex<double> x[], std::complex<double>* X);
	static void ifft(uint32_t n, std::complex<double> X[], std::complex<double>* x);
  static void dft_2D(uint32_t m, uint32_t n, std::complex<double> x[], std::complex<double>* X);
	static void idft_2D(uint32_t m, uint32_t n, std::complex<double> X[], std::complex<double>* x);

  static void pad_kernel(uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double>* pad_ker);
  static inline void pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double>* p);


  Image& fd_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);
  Image& std_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc);

  // void applyNoiseFiltering(); // Applies median filter to reduce noise
  Image applyNoiseFiltering() const;
};

void insertionSort(int arr[], int n);
// int noise_filtering();