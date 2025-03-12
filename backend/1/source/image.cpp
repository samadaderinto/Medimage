#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION

#define BYTE_BOUND(value) value < 0 ? 0 : (value > 255 ? 255 : value)

#include "../include/image.h"
#include "../include/stb_image.h"
#include "../include/stb_image_write.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <complex>

using namespace std;

void insertionSort(int arr[], int n)
{
  int i, key, j;
  for (i = 1; i < n; i++)
  {
    key = arr[i];
    j = i - 1;
    while (j >= 0 && arr[j] > key)
    {
      arr[j + 1] = arr[j];
      j = j - 1;
    }
    arr[j + 1] = key;
  }
}

Image::Image(const char *filename)
{
  if (read(filename))
  {
    printf("Read %s\n", filename);
    size = w * h * channels;
  }
  else
  {
    printf("Failed to read %s\n", filename);
  }
}
Image::Image(int w, int h, int channels) : w(w), h(h), channels(channels)
{
  size = w * h * channels;
  data = new uint8_t[size];
}
Image::Image(const Image &img) : Image(img.w, img.h, img.channels)
{
  memcpy(data, img.data, size);
}
Image::~Image()
{
  // cleans all the data. Call when an image gets destructed.
  stbi_image_free(data);
}

bool Image::read(const char *filename)
{
  data = stbi_load(filename, &w, &h, &channels, 0); // Function will fail if data is null.
  return data != NULL;
}
bool Image::write(const char *filename)
{
  ImageType type = getFileType(filename);
  int success;
  switch (type)
  {
  case PNG:
    success = stbi_write_png(filename, w, h, channels, data, w * channels);
    break;
  case BMP:
    success = stbi_write_bmp(filename, w, h, channels, data);
    break;
  case JPG:
    success = stbi_write_jpg(filename, w, h, channels, data, 100);
    break;
  case TGA:
    success = stbi_write_tga(filename, w, h, channels, data);
    break;
  }
  return success != 0;
}

ImageType Image::getFileType(const char *filename)
{
  const char *ext = strrchr(filename, '.');
  if (ext != nullptr)
  {
    if (strcmp(ext, ".png") == 0)
    {
      return PNG;
    }
    else if (strcmp(ext, ".jpg") == 0)
    {
      return JPG;
    }
    else if (strcmp(ext, ".bmp") == 0)
    {
      return BMP;
    }
    else if (strcmp(ext, ".tga") == 0)
    {
      return TGA;
    }
  }
  return PNG;
}

Image &Image::grayscale_avg()
{
  if (channels < 3)
  {
    printf("Image %p channels is less than 3. It is assumed to be already grayscaled.", this);
  }
  else
  {
    for (int i = 0; i < size; i += channels)
    {
      int gray = (data[i] + data[i + 1] + data[i + 2]) / 3;
      memset(data + i, gray, 3);
    }
  }
  return *this;
}
Image &Image::grayscale_lum()
{
  if (channels < 3)
  {
    printf("Image %p channels is less than 3. It is assumed to be already grayscaled.", this);
  }
  else
  {
    for (int i = 0; i < size; i += channels)
    {
      int gray = (0.2126 * data[i] + 0.7152 * data[i + 1] + 0.0722 * data[i + 2]) / 3;
      memset(data + i, gray, 3);
    }
  }
  return *this;
}

Image &Image::colorMask(float r, float g, float b)
{
  if (channels < 3)
  {
    printf("\e[31m[ERROR] Color mask requires at least 3 channels but this image has %d channels\e[0m\n", channels);
  }
  else
  {
    for (int i = 0; i < size; i += channels)
    {
      data[i] *= r;
      data[i + 1] *= g;
      data[i + 2] *= b;
    }
  }
  return *this;
}

Image &Image::encodeMessage(const char *message)
{
  return *this;
}
Image &Image::decodeMessage(char *buffer, size_t *messageLength)
{
  return *this;
}

// void Image::applyNoiseFiltering()
// {
//   if (data == nullptr)
//   {
//     std::cerr << "Image data is null. Load an image first." << std::endl;
//     return;
//   }

//   uint8_t *filteredData = new uint8_t[w * h * channels](); // New filtered image buffer
//   int window[9];

//   // Apply median filter
//   for (int row = 1; row < h - 1; ++row)
//   {
//     for (int col = 1; col < w - 1; ++col)
//     {
//       for (int c = 0; c < channels; ++c)
//       {
//         // Populate the 3x3 window for the current pixel
//         window[0] = data[((row - 1) * w + (col - 1)) * channels + c];
//         window[1] = data[((row - 1) * w + col) * channels + c];
//         window[2] = data[((row - 1) * w + (col + 1)) * channels + c];
//         window[3] = data[(row * w + (col - 1)) * channels + c];
//         window[4] = data[(row * w + col) * channels + c];
//         window[5] = data[(row * w + (col + 1)) * channels + c];
//         window[6] = data[((row + 1) * w + (col - 1)) * channels + c];
//         window[7] = data[((row + 1) * w + col) * channels + c];
//         window[8] = data[((row + 1) * w + (col + 1)) * channels + c];

//         // Sort the window to find the median
//         insertionSort(window, 9);

//         // Assign the median value to the filtered data
//         filteredData[(row * w + col) * channels + c] = window[4];
//       }
//     }
//   }

//   // Replace old data with the filtered data
//   delete[] data;
//   data = filteredData;
// }

Image &Image::diffmap(Image &img)
{
  // We get a minimum width betwen both images to compare because thie sizes may vary.
  int compare_width = fmin(w, img.w);
  int compare_height = fmin(h, img.h);
  int compare_channels = fmin(channels, img.channels);
  for (uint32_t i = 0; i < compare_height; ++i)
  {
    for (uint32_t j = 0; j < compare_width; ++j)
    {
      for (uint32_t k = 0; k < compare_channels; ++k)
      {
        data[(i * w + j) * channels + k] = BYTE_BOUND(abs(data[(i * w + j) * channels + k] - img.data[(i * img.w + j) * img.channels + k]));
      }
    }
  }
  return *this;
}

Image &Image::diffmap_scale(Image &img, uint8_t scl)
{
  // We get a minimum width betwen both images to compare because thie sizes may vary.
  int compare_width = fmin(w, img.w);
  int compare_height = fmin(h, img.h);
  int compare_channels = fmin(channels, img.channels);
  uint8_t largest = 0;
  for (uint32_t i = 0; i < compare_height; ++i)
  {
    for (uint32_t j = 0; j < compare_width; ++j)
    {
      for (uint32_t k = 0; k < compare_channels; ++k)
      {
        data[(i * w + j) * channels + k] = BYTE_BOUND(abs(data[(i * w + j) * channels + k] - img.data[(i * img.w + j) * img.channels + k]));
        largest = fmax(largest, data[(i * w + j) * channels + k]);
      }
    }
  }
  scl = 255 / fmax(1, fmax(scl, largest));
  // We don't want scale to be lower than the largest pixel value 'cos it might break it.
  for (int i = 0; i < size; i++)
  {
    data[i] *= scl;
  }
  return *this;
}

Image &Image::flipX()
{
  uint8_t tmp[4];
  uint8_t *px2;
  uint8_t *px1;
  for (int y = 0; y < h; y++)
  {
    for (int x = 0; x < w / 2; x++)
    {
      px1 = &data[(x + y * w) * channels];
      px2 = &data[((w - 1 - x) + y * w) * channels];
      memcpy(tmp, px1, channels);
      memcpy(px1, px2, channels);
      memcpy(px2, tmp, channels);
    }
  }
  return *this;
}
Image &Image::flipY()
{
  uint8_t tmp[4];
  uint8_t *px2;
  uint8_t *px1;
  for (int x = 0; x < w; x++)
  {
    for (int y = 0; y < h / 2; y++)
    {
      px1 = &data[(x + y * w) * channels];
      px2 = &data[(x + (h - 1 - y) * w) * channels];
      memcpy(tmp, px1, channels);
      memcpy(px1, px2, channels);
      memcpy(px2, tmp, channels);
    }
  }
  return *this;
}

Image Image::applyNoiseFiltering() const
{
  if (data == nullptr)
  {
    std::cerr << "Image data is null. Load an image first." << std::endl;
    return Image(0, 0, 0); // Return an empty image
  }

  // Create a new image for the filtered data
  Image filteredImage(w, h, channels);
  uint8_t *filteredData = filteredImage.data; // Pointer to the new image's data
  int window[9];

  // Apply median filter
  for (int row = 3; row < h - 3; ++row)
  {
    for (int col = 3; col < w - 3; ++col)
    {
      for (int c = 0; c < channels; ++c)
      {
        // Populate the 3x3 window for the current pixel
        window[0] = data[((row - 1) * w + (col - 1)) * channels + c];
        window[1] = data[((row - 1) * w + col) * channels + c];
        window[2] = data[((row - 1) * w + (col + 1)) * channels + c];
        window[3] = data[(row * w + (col - 1)) * channels + c];
        window[4] = data[(row * w + col) * channels + c];
        window[5] = data[(row * w + (col + 1)) * channels + c];
        window[6] = data[((row + 1) * w + (col - 1)) * channels + c];
        window[7] = data[((row + 1) * w + col) * channels + c];
        window[8] = data[((row + 1) * w + (col + 1)) * channels + c];

        // Sort the window to find the median
        insertionSort(window, 9);

        // Assign the median value to the filtered data
        filteredData[(row * w + col) * channels + c] = window[4];
      }
    }
  }

  return filteredImage;
}

Image &Image::crop(uint16_t cx, uint16_t cy, uint16_t cw, uint16_t ch)
/*
Args
  cx => x start coordinate
  cy => y start coordinate
  cw => Width of the image to be cropped out. (End x coordinate)
  ch => height of the image to be cropped out. (End y coordinate)
*/
{
  size = cw * ch * channels; // Size of 1d array required
  uint8_t* croppedImage = new uint8_t[size]; // Defined 1d array for efficient image processing.

  memset(croppedImage, 0, size); // Set all pixel color to black. 

  for(uint16_t y = 0; y < ch; ++y) {
    if (y + cy >= h) {break;} // Avoids going beyong image boundary and encountering segfault.
    for(uint16_t x = 0; x < cw; ++x) {
      if (x + cx >= w) {break;} // Avoids going beyong image boundary and encountering segfault.
      memcpy(&croppedImage[(x + y * cw) * channels], &data[(x + cx + (y + cy) * w) * channels], channels); // Copy pixel values from data array to cropped image array.
    }
  }

  w = cw;
  h = ch;
  size = w * h * channels;

  delete[] data;
  data = croppedImage;
  croppedImage = nullptr;
  return *this;
}

void Image::dft_2D(uint32_t m, uint32_t n, std::complex<double> x[], std::complex<double>* X) {
	//x in row-major & standard order
	std::complex<double>* intermediate = new std::complex<double>[m*n];
	//rows
	for(uint32_t i=0; i<m; ++i) {
		fft(n, x+i*n, intermediate+i*n);
	}
	//cols
	for(uint32_t j=0; j<n; ++j) {
		for(uint32_t i=0; i<m; ++i) {
			X[j*m+i] = intermediate[i*n+j]; //row-major --> col-major
		}
		fft(m, X+j*m, X+j*m);
	}
	delete[] intermediate;
	//X in column-major & bit-reversed (in rows then columns)
}

void Image::pointwise_product(uint64_t l, std::complex<double> a[], std::complex<double> b[], std::complex<double>* p) {
	for(uint64_t k=0; k<l; ++k) {
		p[k] = a[k]*b[k];
	}
}

void Image::fft(uint32_t n, std::complex<double> x[], std::complex<double>* X) {
	//x in standard order
	if(x != X) {
		memcpy(X, x, n*sizeof(std::complex<double>));
	}

	//Gentleman-Sande butterfly
	uint32_t sub_probs = 1;
	uint32_t sub_prob_size = n;
	uint32_t half;
	uint32_t i;
	uint32_t j_begin;
	uint32_t j_end;
	uint32_t j;
	std::complex<double> w_step;
	std::complex<double> w;
	std::complex<double> tmp1, tmp2;
	while(sub_prob_size>1) {
		half = sub_prob_size>>1;
		w_step = std::complex<double>(cos(-2*M_PI/sub_prob_size), sin(-2*M_PI/sub_prob_size));
		for(i=0; i<sub_probs; ++i) {
			j_begin = i*sub_prob_size;
			j_end = j_begin+half;
			w = std::complex<double>(1,0);
			for(j=j_begin; j<j_end; ++j) {
				tmp1 = X[j];
				tmp2 = X[j+half];
				X[j] = tmp1+tmp2;
				X[j+half] = (tmp1-tmp2)*w;
				w *= w_step;
			}
		}
		sub_probs <<= 1;
		sub_prob_size = half;
	}
	//X in bit reversed order
}

void Image::ifft(uint32_t n, std::complex<double> X[], std::complex<double>* x) {
	//X in bit reversed order
	if(X != x) {
		memcpy(x, X, n*sizeof(std::complex<double>));
	}

	//Cooley-Tukey butterfly
	uint32_t sub_probs = n>>1;
	uint32_t sub_prob_size;
	uint32_t half = 1;
	uint32_t i;
	uint32_t j_begin;
	uint32_t j_end;
	uint32_t j;
	std::complex<double> w_step;
	std::complex<double> w;
	std::complex<double> tmp1, tmp2;
	while(half<n) {
		sub_prob_size = half<<1;
		w_step = std::complex<double>(cos(2*M_PI/sub_prob_size), sin(2*M_PI/sub_prob_size));
		for(i=0; i<sub_probs; ++i) {
			j_begin = i*sub_prob_size;
			j_end = j_begin+half;
			w = std::complex<double>(1,0);
			for(j=j_begin; j<j_end; ++j) {
				tmp1 = x[j];
				tmp2 = w*x[j+half];
				x[j] = tmp1+tmp2;
				x[j+half] = tmp1-tmp2;
				w *= w_step;
			}
		}
		sub_probs >>= 1;
		half = sub_prob_size;
	}
	for(uint32_t i=0; i<n; ++i) {
		x[i] /= n;
	}
	//x in standard order
}

void Image::idft_2D(uint32_t m, uint32_t n, std::complex<double> X[], std::complex<double>* x) {
	//X in column-major & bit-reversed (in rows then columns)
	std::complex<double>* intermediate = new std::complex<double>[m*n];
	//cols
	for(uint32_t j=0; j<n; ++j) {
		ifft(m, X+j*m, intermediate+j*m);
	}
	//rows
	for(uint32_t i=0; i<m; ++i) {
		for(uint32_t j=0; j<n; ++j) {
			x[i*n+j] = intermediate[j*m+i]; //row-major <-- col-major
		}
		ifft(n, x+i*n, x+i*n);
	}
	delete[] intermediate;
	//x in row-major & standard order
}

void Image::pad_kernel(uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc, uint32_t pw, uint32_t ph, std::complex<double>* pad_ker) {
	//padded so center of kernel is at top left
	for(long i=-((long)cr); i<(long)ker_h-cr; ++i) {
		uint32_t r = (i<0) ? i+ph : i;
		for(long j=-((long)cc); j<(long)ker_w-cc; ++j) {
			uint32_t c = (j<0) ? j+pw : j;
			pad_ker[r*pw+c] = std::complex<double>(ker[(i+cr)*ker_w+(j+cc)], 0);
		}
	}
}

Image& Image::fd_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	//calculate padding
	uint32_t pw = 1<<((uint8_t)ceil(log2(w+ker_w-1)));
	uint32_t ph = 1<<((uint8_t)ceil(log2(h+ker_h-1)));
	uint64_t psize = pw*ph;

	//pad image
	std::complex<double>* pad_img = new std::complex<double>[psize];
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			pad_img[i*pw+j] = std::complex<double>(data[(i*w+j)*channels+channel],0);
		}
	}

	//pad kernel
	std::complex<double>* pad_ker = new std::complex<double>[psize];
	pad_kernel(ker_w, ker_h, ker, cr, cc, pw, ph, pad_ker);

	//convolution
	dft_2D(ph, pw, pad_img, pad_img);
	dft_2D(ph, pw, pad_ker, pad_ker);
	pointwise_product(psize, pad_img, pad_ker, pad_img);
	idft_2D(ph, pw, pad_img, pad_img);

	//update pixel data
	for(uint32_t i=0; i<h; ++i) {
		for(uint32_t j=0; j<w; ++j) {
			data[(i*w+j)*channels+channel] = BYTE_BOUND((uint8_t)round(pad_img[i*pw+j].real()));
		}
	}

	return *this;
}

Image& Image::std_convolve_clamp_to_0(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	// uint8_t new_data[w*h];
  uint8_t* new_data = new uint8_t[w * h];
  // std::vector<uint8_t> new_data(w * h);
	uint64_t center = cr*ker_w + cc;
	for(uint64_t k=channel; k<size; k+=channels) {
		double c = 0;
		for(long i = -((long)cr); i<(long)ker_h-cr; ++i) {
			long row = ((long)k/channels)/w-i;
			if(row < 0 || row > h-1) {
				continue;
			}
			for(long j = -((long)cc); j<(long)ker_w-cc; ++j) {
				long col = ((long)k/channels)%w-j;
				if(col < 0 || col > w-1) {
					continue;
				}
				c += ker[center+i*(long)ker_w+j]*data[(row*w+col)*channels+channel];
			}
		}
		new_data[k/channels] = (uint8_t)BYTE_BOUND(round(c));
	}
	for(uint64_t k=channel; k<size; k+=channels) {
		data[k] = new_data[k/channels];
	}
	return *this;
}


Image& Image::convolve_linear(uint8_t channel, uint32_t ker_w, uint32_t ker_h, double ker[], uint32_t cr, uint32_t cc) {
	if(ker_w*ker_h > 224) {
		return fd_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	}
	else {
		return std_convolve_clamp_to_0(channel, ker_w, ker_h, ker, cr, cc);
	}
}