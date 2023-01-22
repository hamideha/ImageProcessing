import cv2
import numpy as np
import bilinear

def compress(img):
  img = cv2.imread(img, 1)
  
  B, G, R = cv2.split(img)
  
  Y = 0.299*R + 0.587*G + 0.114*B
  U = -0.14713*R - 0.288862*G + 0.436*B
  V = 0.615*R - 0.51498*G - 0.10001*B

  # Downsample Y by a factor of 2
  Y = Y[::2, ::2]
  
  # Downsample U and V by a factor of 4
  U = U[::4, ::4]
  V = V[::4, ::4]

  Y = np.clip(Y, 0, 255).astype(np.uint8)
  U = np.clip(U, 0, 255).astype(np.uint8)
  V = np.clip(V, 0, 255).astype(np.uint8)

  return (Y, U, V)

def reconstruct(Y, U, V):
  m, n = Y.shape

  Y = bilinear(Y, m, n)
  U = bilinear(U, m, n)
  V = bilinear(V, m, n)

  R = Y + 1.13983*V
  G = Y - 0.39465*U - 0.5806*V
  B = Y + 2.03211*U

  R = np.clip(R, 0, 255).astype(np.uint8)
  G = np.clip(G, 0, 255).astype(np.uint8)
  B = np.clip(B, 0, 255).astype(np.uint8)

  img = cv2.merge((B, G, R))
  return img

def main():
  Y, U, V = compress('Bird.jpg')
  img_u = reconstruct(Y, U, V)

  cv2.imshow('Y', img_u)
  cv2.waitKey(0) 
  cv2.destroyAllWindows()

if __name__ == '__main__':
  main()  
