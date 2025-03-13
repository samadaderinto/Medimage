#include <iostream>
#include <fstream>
#include <vector>
#include <openssl/evp.h>
#include <openssl/rand.h>

//const unsigned char AES_USER_KEY[16] = "1234567890abcdef"; // Fixed variable name
//const unsigned char AES_IV[16] = "1234567890abcdef";      // Initialization Vector (IV)
const unsigned char AES_USER_KEY[16] = { 
    '1', '2', '3', '4', '5', '6', '7', '8', 
    '9', '0', 'a', 'b', 'c', 'd', 'e', 'f' 
}; 

const unsigned char AES_IV[16] = { 
    'f', 'e', 'd', 'c', 'b', 'a', '0', '9', 
    '8', '7', '6', '5', '4', '3', '2', '1' 
};

// Function to encrypt data
void encryptImage(const std::vector<unsigned char>& inputData, std::vector<unsigned char>& encryptedData) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len;
    int ciphertext_len;

    encryptedData.resize(inputData.size() + 16); // Extra space for padding

    EVP_EncryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, AES_USER_KEY, AES_IV);
    EVP_EncryptUpdate(ctx, encryptedData.data(), &len, inputData.data(), inputData.size());
    ciphertext_len = len;
    EVP_EncryptFinal_ex(ctx, encryptedData.data() + len, &len);
    ciphertext_len += len;

    encryptedData.resize(ciphertext_len); // Trim excess space
    EVP_CIPHER_CTX_free(ctx);
}

// Function to decrypt data
void decryptImage(const std::vector<unsigned char>& encryptedData, std::vector<unsigned char>& decryptedData) {
    EVP_CIPHER_CTX *ctx = EVP_CIPHER_CTX_new();
    int len;
    int plaintext_len;

    decryptedData.resize(encryptedData.size());

    EVP_DecryptInit_ex(ctx, EVP_aes_128_cbc(), NULL, AES_USER_KEY, AES_IV);
    EVP_DecryptUpdate(ctx, decryptedData.data(), &len, encryptedData.data(), encryptedData.size());
    plaintext_len = len;
    EVP_DecryptFinal_ex(ctx, decryptedData.data() + len, &len);
    plaintext_len += len;

    decryptedData.resize(plaintext_len); // Trim excess space
    EVP_CIPHER_CTX_free(ctx);
}

// Function to read image file
std::vector<unsigned char> readImageFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

// Function to write encrypted/decrypted file
void writeImageFile(const std::string& filename, const std::vector<unsigned char>& data) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(data.data()), data.size());
}

int main() {
    std::string inputFilename = "thyroid_ultrasound.jpg";
    std::string encryptedFilename = "encrypted_thyroid.dat";
    std::string decryptedFilename = "decrypted_thyroid.jpg";

    // Read the input image
    std::vector<unsigned char> imageData = readImageFile(inputFilename);
    std::vector<unsigned char> encryptedData, decryptedData;

    // Encrypt and save
    encryptImage(imageData, encryptedData);
    writeImageFile(encryptedFilename, encryptedData);
    std::cout << "Image encrypted and saved as: " << encryptedFilename << std::endl;

    // Decrypt and save
    decryptImage(encryptedData, decryptedData);
    writeImageFile(decryptedFilename, decryptedData);
    std::cout << "Image decrypted and saved as: " << decryptedFilename << std::endl;

    return 0;
}
