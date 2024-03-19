
#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include "tensor.h"
#include <math.h>

void Prediction(float image[28][28],
                float w_conv1[6][1][1],
                float w_conv2[16][6][5][5],
                float w_fc1[120][400],
                float w_fc2[84][120],
                float w_fc3[10][84],
                float b_conv1[6],
                float b_conv2[16],
                float b_fc1[120],
                float b_fc2[84],
                float b_fc3[10],
                float probs[10]) {

    // Intermediate variables
    float conv1_out[6][28][28];
    float pool1_out[6][14][14];
    float conv2_out[16][10][10];
    float pool2_out[16][5][5];
    float fc1_out[120];
    float fc2_out[84];
    float fc3_out[10];
    int i, j, m, n, p, q, k;

    // Convolution 1
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 28; j++) {
            for (k = 0; k < 28; k++) {
                conv1_out[i][j][k] = 0;
                for (m = 0; m < 1; m++) {
                    for (n = 0; n < 1; n++) {
                        conv1_out[i][j][k] += image[j + m][k + n] * w_conv1[i][m][n];
                    }
                }
                conv1_out[i][j][k] += b_conv1[i];
                conv1_out[i][j][k] = fmaxf(0, conv1_out[i][j][k]); // ReLU activation
            }
        }
    }

    // Pooling 1 (2x2 Max pooling)
    for (i = 0; i < 6; i++) {
        for (j = 0; j < 14; j++) {
            for (k = 0; k < 14; k++) {
                float max_val = conv1_out[i][2*j][2*k];
                for (m = 0; m < 2; m++) {
                    for (n = 0; n < 2; n++) {
                        if (conv1_out[i][2*j+m][2*k+n] > max_val) {
                            max_val = conv1_out[i][2*j+m][2*k+n];
                        }
                    }
                }
                pool1_out[i][j][k] = max_val;
            }
        }
    }

    // Convolution 2
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 10; j++) {
            for (k = 0; k < 10; k++) {
                conv2_out[i][j][k] = 0;
                for (m = 0; m < 6; m++) {
                    for (n = 0; n < 5; n++) {
                        for (p = 0; p < 5; p++) {
                            conv2_out[i][j][k] += pool1_out[m][j + n][k + p] * w_conv2[i][m][n][p];
                        }
                    }
                }
                conv2_out[i][j][k] += b_conv2[i];
                conv2_out[i][j][k] = fmaxf(0, conv2_out[i][j][k]); // ReLU activation
            }
        }
    }

    // Pooling 2 (2x2 Max pooling)
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 5; j++) {
            for (k = 0; k < 5; k++) {
                float max_val = conv2_out[i][2*j][2*k];
                for (m = 0; m < 2; m++) {
                    for (n = 0; n < 2; n++) {
                        if (conv2_out[i][2*j+m][2*k+n] > max_val) {
                            max_val = conv2_out[i][2*j+m][2*k+n];
                        }
                    }
                }
                pool2_out[i][j][k] = max_val;
            }
        }
    }

    // Flatten for Fully Connected layers
    float flatten[400];
    for (i = 0; i < 16; i++) {
        for (j = 0; j < 5; j++) {
            for (k = 0; k < 5; k++) {
                flatten[25*i + 5*j + k] = pool2_out[i][j][k];
            }
        }
    }

    // Fully Connected 1
    for (i = 0; i < 120; i++) {
        fc1_out[i] = 0;
        for (j = 0; j < 400; j++) {
            fc1_out[i] += flatten[j] * w_fc1[i][j];
        }
        fc1_out[i] += b_fc1[i];
        fc1_out[i] = fmaxf(0, fc1_out[i]); // ReLU activation
    }

    // Fully Connected 2
    for (i = 0; i < 84; i++) {
        fc2_out[i] = 0;
        for (j = 0; j < 120; j++) {
            fc2_out[i] += fc1_out[j] * w_fc2[i][j];
        }
        fc2_out[i] += b_fc2[i];
        fc2_out[i] = fmaxf(0, fc2_out[i]); // ReLU activation
    }

    // Fully Connected 3
    for (i = 0; i < 10; i++) {
        fc3_out[i] = 0;
        for (j = 0; j < 84; j++) {
            fc3_out[i] += fc2_out[j] * w_fc3[i][j];
        }
        fc3_out[i] += b_fc3[i];
    }

    // Softmax activation
    float sum_exp = 0;
    for (i = 0; i < 10; i++) {
        probs[i] = expf(fc3_out[i]);
        sum_exp += probs[i];
    }
    for (i = 0; i < 10; i++) {
        probs[i] /= sum_exp;
    }
}

int main(int argc, char** argv){

   //float image[28][28];
   float w_conv1[6][1][1];
   float w_conv2[16][6][5][5];
   float w_fc1[120][400];
   float w_fc2[84][120];
   float w_fc3[10][84];
   float b_conv1[6];
   float b_conv2[16];
   float b_fc1[120];
   float b_fc2[84];
   float b_fc3[10];
   float probs[10];

   int i,j,m,n,index;
   FILE *fp;

    /* Load Weights from DDR->LMM */
   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/w_conv1.txt", "r");
   for(i=0;i<6;i++)
       fscanf(fp, "%f ",  &(w_conv1[i][0][0]));  fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/w_conv2.txt", "r");
   for(i=0;i<16;i++){
       for(j=0;j<6;j++){
           for(m=0;m<5;m++){
               for(n=0;n<5;n++){
                   index = 16*i + 6*j + 5*m + 5*n;
                   fscanf(fp, "%f ",  &(w_conv2[i][j][m][n]));
               }
           }
       }
   }
   fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/w_fc1.txt", "r");
   for(i=0;i<120;i++){
       for(j=0;j<400;j++)
           fscanf(fp, "%f ",  &(w_fc1[i][j]));
   }
   fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/w_fc2.txt", "r");
   for(i=0;i<84;i++){
       for(j=0;j<120;j++)
           fscanf(fp, "%f ",  &(w_fc2[i][j]));
   }
   fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/w_fc3.txt", "r");
   for(i=0;i<10;i++){
       for(j=0;j<84;j++)
           fscanf(fp, "%f ",  &(w_fc3[i][j]));
   }
   fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/b_conv1.txt", "r");
   for(i=0;i<6;i++)
       fscanf(fp, "%f ",  &(b_conv1[i]));  fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/b_conv2.txt", "r");
   for(i=0;i<16;i++)
       fscanf(fp, "%f ",  &(b_conv2[i]));  fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/b_fc1.txt", "r");
   for(i=0;i<120;i++)
       fscanf(fp, "%f ",  &(b_fc1[i]));  fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/b_fc2.txt", "r");
   for(i=0;i<84;i++)
       fscanf(fp, "%f ",  &(b_fc2[i]));  fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/data/weights/b_fc3.txt", "r");
   for(i=0;i<10;i++)
       fscanf(fp, "%f ",  &(b_fc3[i]));  fclose(fp);

   float *dataset = (float*)malloc(LABEL_LEN*28*28 *sizeof(float));
   int target[LABEL_LEN];

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/mnist-test-target.txt", "r");
   for(i=0;i<LABEL_LEN;i++)
       fscanf(fp, "%d ",  &(target[i]));  fclose(fp);

   fp = fopen("D:/DTVT_KHTN/THXLA/LeDinhHuy-TonDucPhuVinh-THXLA-Lab4/cnn_lenet5/mnist-test-image.txt", "r");
   for(i=0;i<LABEL_LEN*28*28;i++)
       fscanf(fp, "%f ",  &(dataset[i]));  fclose(fp);

   float image[28][28];
   float *datain;
   int acc = 0;
   int mm, nn;
   for(i=0;i<LABEL_LEN;i++) {

       datain = &dataset[i*28*28];
       for(mm=0;mm<28;mm++)
           for(nn=0;nn<28;nn++)
               image[mm][nn] = *(float*)&datain[28*mm + nn];

       Prediction(  image,
                    w_conv1,
                    w_conv2,
                    w_fc1,
                    w_fc2,
                    w_fc3,
                    b_conv1,
                    b_conv2,
                    b_fc1,
                    b_fc2,
                    b_fc3,
                    probs
                    );

       int index = 0;
       float max = probs[0];
       for (j=1;j<10;j++) {
            if (probs[j] > max) {
                index = j;
                max = probs[j];
            }
       }

       if (index == target[i]) acc++;
       printf("Predicted label: %d\n", index);
       printf("Prediction: %d/%d\n", acc, i+1);
   }
   printf("Accuracy = %f\n", acc*1.0f/LABEL_LEN);

    return 0;
}


