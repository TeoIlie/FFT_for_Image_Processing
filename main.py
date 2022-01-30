#!/usr/bin/python3
#
# Removal of periodic features using the FFT
#
# Use Python 3+ with these packages: numpy, PyOpenGL, Pillow, glfw
#
# DO NOT IMPORT OR USE ANY ADDITIONAL LIBRARIES.


import sys, os, math, statistics

import matplotlib.pyplot as plt

try: # NumPy
  import numpy as np
except:
  print( 'Error: NumPy has not been installed.' )
  sys.exit(0)

try: # Pillow
  from PIL import Image, ImageOps
except:
  print( 'Error: Pillow has not been installed.' )
  sys.exit(0)

try: # PyOpenGL
  from OpenGL.GLUT import *
  from OpenGL.GL import *
  from OpenGL.GLU import *
except:
  print( 'Error: PyOpenGL has not been installed.' )
  sys.exit(0)

try: # GLFW
  import glfw
except:
  print( 'Error: GLFW has not been installed.' )
  sys.exit(0)


# Globals

windowWidth  = 1000 # window dimensions (not image dimensions)
windowHeight =  800

showMagnitude = True            # for the FT, show the magnitude.  Otherwise, show the phase
doHistoEq = False               # do histogram equalization on the FT to make features more obvious

texID = None                    # for OpenGL

zoom = 1.0                      # amount by which to zoom images
translate = (0.0,0.0)           # amount by which to translate images


# Image

imageDir      = 'images'
imageFilename = 'small.png'
imagePath     = os.path.join( imageDir, imageFilename )

image    = None                 # the image as a 2D np.array
image_FT  = None                 # the image's FT as a 2D np.array

gridImage   = None              # the grid, isolated from the image
gridImage_FT = None              # the grid's FT

resultImage = None              # the final image


# Remove the grid from the global 'image'.  Return the result image
# AND a list of [ [angle1,distance1], [angle2,distance2] ] describing
# the two principal grid lines.
#
# The angle is the angle, in degrees, of the grid line
# counterclockwise from the horizontal.
#
# The distance is the distance from the origin, in pixels, of the
# first peak in the Fourier Transform corresponding to the lines at
# the given angle.  This will later be used to calculate the line
# spacing.
#
# Do the following in the compute() function:
#
#   1. Compute the FT of the image.  Store it in 'image_FT'.
#
#   2. Compute and store the FT magnitudes.  Find the maximum
#      magnitude, EXCLUDING the DC component in [0,0]. 
#
#   3. Set to zero the components of 'image_FT' that have magnitude
#      less than 40% the maximum magnitude.  Store this new FT in
#      'gridImage_FT'.  Record in a list the (x,y) locations of the
#      non-zero magnitudes of 'gridImage_FT'.
#
#   4. From the locations of the non-zero magnitudes, find the angles
#      of the two principal grid lines and, for each such line, find
#      the distance of the closest non-zero magnitude to the origin.
#
#      THIS IS DIFFICULT and can be left until you have zeroed the grid
#      line pixels as described in Step 6.
#
#   5. Apply the inverse FT to 'gridImage_FT' to get 'gridImage'.
#
#   6. For each (x,y) location in 'gridImage' that has a bright pixel
#      of value > 16 (i.e. is one of the grid lines), set the
#      corresponding pixel in the original 'image' to the average of
#      the pixels on either side of the grid line that are not also
#      grid line pixels.  Do not modify 'image'; instead, store your
#      result in 'resultImage'.
#
#      FINDING THE AVERAGE IS DIFFICULT, so first set the grid line
#      pixels to zero and debug your code to make sure that this is
#      working.  Only after this is working should you try to set the
#      grid line pixels to the average from either side.  You will need 
#      to know the angles from Step 4 to do this.
#       - limit search to 3 pixels in each direction 
#       - use gradient, or search along both grid line directions 
#
# Test your code on all files in the 'images' directory.  Debug using the 'small.png' file.
#
# DO NOT IMPORT OR USE ANY ADDITIONAL LIBRARIES.
#
# DO NOT USE NUMPY FOR ANYTHING BUT THE FOLLOWING OPERATIONS:
#     
#     np.absolute
#     np.complex_
#     np.imag
#     np.max
#     np.median
#     np.real
#     np.sqrt
#     np.sqrt
#     np.tan
#     np.vectorize
#     np.zeros
#
# In the compute() function below, you should always iterate over the
# image in your own code, rather than call some NumPy function to do
# the iteration for you.


def compute():

  global image, image_FT, gridImage, gridImage_FT, resultImage

  height = image.shape[0]
  width  = image.shape[1]

  # Forward FT

  print ('1. compute FT')
  image_FT = forwardFT(image)
      
  # Compute magnitudes and find the maximum (excluding the DC component)

  print ('2. computing FT magnitudes')
  # calculate a_k and b_k using real and imaginary components
  # take euclidean for magnitude, and store 
  a_k = 2 * np.real(image_FT)
  b_k = -2 * np.imag(image_FT)

  magnitudes = np.sqrt( a_k*a_k + b_k*b_k )

  # remove centre and find maximum magnitude
  magnitudes_flat = np.delete(magnitudes.flatten(), 0)
  largest_mag = np.max(magnitudes_flat)

  # zero the components that are less than 40% of the max

  print ('3. removing low-magnitude components')

  if gridImage_FT is None:
    gridImage_FT = np.zeros( (height,width), dtype=np.complex_ )
    
  coords = []

  # discard all components that are less than 40% of largest magnitude
  for i in range(height):
    for j in range(width):

      if (magnitudes[i][j] < (0.4 * largest_mag)):
        # set components less than 40% of largest magnitude to 0
        gridImage_FT[i][j] = 0
      else:
        gridImage_FT[i][j] = image_FT[i][j]
        coords.append((j,i))

  # Find (angle, distance) to each peak
  # lines = [ (angle1,distance1), (angle2,distance2) ]

  print( '4. finding angles and distances of grid lines' )

  # 
  norm_coords = []
  angles = []
  u_distances = []
  v_distances = []
  for e in coords:
    u = wrap(e[0] - width/2, width) - width/2
    v = (wrap(e[1] - height/2, height) - height/2)
    distance = math.sqrt(u*u + v*v)
    # print(distance)
    if abs(u) > abs(v):
      u_distances.append(distance)
    elif abs(v) > abs(u):
      v_distances.append(distance)
    # normalize v
    v *= width/height

    if u is not 0:
      norm_coords.append((u,v))
      x = math.degrees(math.atan2(v,u))
      if (x < 0):
        x = 180 + x
      angles.append(x)
  u_distances.sort()
  v_distances.sort()
  distance1 = round(u_distances[0], 1)
  distance2 = round(v_distances[0], 1)

  angles.sort()

  epsilon = 0.5
  clusters = [[], []]
  in_cluster = False
  clusterCount = -1
  for j in range(len(angles)-2):
    # if next number is close enough
    if angles[j+1] - angles[j] < epsilon:
      # if not in cluster yet, and next 2 numbers are close
      if in_cluster == False and angles[j+2] - angles[j+1] < epsilon:
        # start new cluster
        in_cluster = True
        clusterCount += 1
        if clusterCount > 1:
          break
        clusters[clusterCount].append(angles[j])
      # append number to cluster
      if in_cluster == True:
        clusters[clusterCount].append(angles[j+1])
    else:
      in_cluster = False

  angle1 = round(statistics.mean(clusters[0]), 1)
  angle2 = round(statistics.mean(clusters[1]), 1)

  lines = [ (angle1,distance1), (angle2,distance2) ]
  
  # Convert back to spatial domain to get a grid-like image

  print ('5. inverse FT')

  if gridImage is None:
    gridImage = np.zeros( (height,width), dtype=np.complex_ )

  # compute inverse Fourier Transform of gridImage and store in gridImage

  gridImage = inverseFT(gridImage_FT)

  # Remove grid image from original image

  print ('6. remove grid')

  if resultImage is None:
    resultImage = image.copy()
  
  # look through gridImage to find grid pixels, and remove them from image
  # to put into resultImage. do this by searching perpendicular to both grid lines, and for
  # the one that finds useful pixels, take average on either side.

  # finding angles perpendicular to grid lines
  perpendicular_1 = wrap((angle1 + 90), 180)
  perpendicular_2 = wrap((angle2 + 90), 180)
  
  # search through gridImage 
  for v in range(width):
    for u in range(height):

      if gridImage[u][v] > 16:
        # pixel values greater than 16 are considered part of a grid line, and must be removed

        # search along both perpendiculars, and pick the correct one
        # begin by searching perpendicular_1
        pixel_1_coord, pixel_2_coord = search_perpendicular(gridImage, perpendicular_1, u, v, height, width)

        # if we did not get values for both pixel coordinates, we try using perpendicular_2
        if pixel_1_coord is None or pixel_2_coord is None:
          pixel_1_coord, pixel_2_coord = search_perpendicular(gridImage, perpendicular_2, u, v, height, width)

        # if still pixels are None we set them to 0 (black)
        if pixel_1_coord is None:
          pixel_1_val = 0
        else:
          pixel_1_val = image[pixel_1_coord[0]][pixel_1_coord[1]]

        if pixel_2_coord is None:
          pixel_2_val = 0
        else:
          pixel_2_val = image[pixel_2_coord[0]][pixel_2_coord[1]]

        # take average in image using coordinates pixel_1, and pixel_2
        pixel_val = (pixel_1_val + pixel_2_val) / 2

        # put that value into resultImage
        resultImage[u][v] = pixel_val

  print ('done')

  return resultImage, lines


# this function takes as input the gridImage and a perpendicular angle, and returns 
# 2 pixel coordinates that are <= 16 in gridImage along the perpendicular angle direction.
# they pixels are None if no such pixel was found at max 3 distance away
def search_perpendicular(gridImage, perpendicular_angle, u, v, height, width):
  pixel_1 = None
  pixel_2 = None
  
  # initiliaze distance, and temp_u, temp_v
  d = 0

  temp_u = round(u + d * math.cos(perpendicular_angle))
  temp_v = round(v + d * math.sin(perpendicular_angle))

  # first search in positive direction along perpendicular_1 line
  while temp_u < (height - 1) and temp_v < (width - 1) and gridImage[temp_u][temp_v] > 16 and d <= 3:
    temp_u = round(u + abs(d * math.cos(perpendicular_angle)))
    temp_v = round(v + abs(d * math.sin(perpendicular_angle)))
    
    d += 1

  if temp_u < width and temp_v < height and gridImage[temp_u][temp_v] < 16:
    pixel_1 = (temp_u, temp_v)

  # reset distance
  d = 0

  # search in negative direction along perpendicular_1 line
  while temp_u >= 0 and temp_v >= 0 and gridImage[temp_u][temp_v] > 16 and d <= 3:
    temp_u = round(u - abs(d * math.cos(perpendicular_angle)))
    temp_v = round(v - abs(d * math.sin(perpendicular_angle)))

    d += 1
  
  if temp_u >= 0 and temp_v >= 0 and gridImage[temp_u][temp_v] < 16:
    pixel_2 = (temp_u, temp_v)

  # print('angle', perpendicular_angle, '(u,v) %.1f %.1f' % (u, v), "temp_u %.1f temp_v %.1f" % (temp_u, temp_v))

  return pixel_1, pixel_2


# File dialog

haveTK = False

if haveTK:
    import Tkinter, tkFileDialog
    root = Tkinter.Tk()
    root.withdraw()



# Do a forward FT
#
# Input is a 2D numpy array of complex values.
# Output is the same.

def forwardFT( image ):

  return np.fft.fft2( image )



# Do an inverse FT
#
# Input is a 2D numpy array of complex values.
# Output is the same.


def inverseFT( image ):

  return np.fft.ifft2( image )



# Set up the display and draw the current image


def display( window ):

  # Clear window

  glClearColor ( 1, 1, 1, 0 )
  glClear( GL_COLOR_BUFFER_BIT )

  glMatrixMode( GL_PROJECTION )
  glLoadIdentity()

  glMatrixMode( GL_MODELVIEW )
  glLoadIdentity()
  glOrtho( 0, windowWidth, 0, windowHeight, 0, 1 )

  # Set up texturing

  global texID
  
  if texID == None:
    texID = glGenTextures(1)

  glPixelStorei( GL_UNPACK_ALIGNMENT, 1 )
  glBindTexture( GL_TEXTURE_2D, texID )

  glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
  glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
  glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, [1,0,0,1] );

  # Images to draw, in rows and columns

  toDraw, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing = getImagesInfo()

  for r in range(rows):
    for c in range(cols):
      if toDraw[r][c] is not None:

        if r == 0: # normal image in row 0 
          img = toDraw[r][c]
        else: # FT in column 1
          img = np.fft.fftshift( toDraw[r][c] ) # shift FT so that origin is in centre (just for display)

        height = scale * img.shape[0]
        width  = scale * img.shape[1]

        # Find lower-left corner

        baseX = (horizSpacing + maxWidth ) * c + horizSpacing
        baseY = (vertSpacing  + maxHeight) * (rows-1-r) + vertSpacing

        # Get pixels and draw

        if r == 0: # for images (in row 0), show the real part of each pixel
          show = np.real(img)
        else: # for FT (in column 1), show magnitude or phase
          ak =  2 * np.real(img)
          bk = -2 * np.imag(img)
          if showMagnitude:
            show = np.log( 1 + np.sqrt( ak*ak + bk*bk ) ) # take the log because there are a few very large values (e.g. the DC component)
          else:
            show = np.arctan2( -1 * bk, ak )

          if doHistoEq and c > 0:
            show = histoEq( show ) # optionally, perform histogram equalization on FT image (but this takes time!)

        # Put the image into a texture, then draw it

        max = show.max()
        min = show.min()
        if max == min:
          max = min+1
          
        imgData = np.array( (np.ravel(show) - min) / (max - min) * 255, np.uint8 )

        glTexImage2D( GL_TEXTURE_2D, 0, GL_INTENSITY, img.shape[1], img.shape[0], 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, imgData )

        # Include zoom and translate

        cx     = 0.5 - translate[0]/width
        cy     = 0.5 - translate[1]/height
        offset = 0.5 / zoom

        glEnable( GL_TEXTURE_2D )

        glBegin( GL_QUADS )
        glTexCoord2f( cx-offset, cy-offset )
        glVertex2f( baseX, baseY )
        glTexCoord2f( cx+offset, cy-offset )
        glVertex2f( baseX+width, baseY )
        glTexCoord2f( cx+offset, cy+offset )
        glVertex2f( baseX+width, baseY+height )
        glTexCoord2f( cx-offset, cy+offset )
        glVertex2f( baseX, baseY+height )
        glEnd()

        glDisable( GL_TEXTURE_2D )

        if zoom != 1 or translate != (0,0):
          glColor3f( 0.8, 0.8, 0.8 )
          glBegin( GL_LINE_LOOP )
          glVertex2f( baseX, baseY )
          glVertex2f( baseX+width, baseY )
          glVertex2f( baseX+width, baseY+height )
          glVertex2f( baseX, baseY+height )
          glEnd()

  # Draw image captions

  glColor3f( 0.2, 0.5, 0.7 )
 
  if image is not None:
    baseX = horizSpacing
    baseY = (vertSpacing  + maxHeight) * (rows) + 8
    drawText( baseX, baseY, imageFilename )

  if image_FT is not None:
    baseX = horizSpacing
    baseY = (vertSpacing  + maxHeight) * (rows-2) + vertSpacing - 18
    drawText( baseX, baseY, 'FT of %s' % imageFilename )

  if gridImage is not None:
    baseX = (horizSpacing + maxWidth) * 1 + horizSpacing
    baseY = (vertSpacing  + maxHeight) * rows + 8
    drawText( baseX, baseY, 'extracted grid' )

  if gridImage_FT is not None:
    baseX = (horizSpacing + maxWidth) * 1 + horizSpacing
    baseY = (vertSpacing  + maxHeight) * (rows-2) + vertSpacing - 18
    drawText( baseX, baseY, 'FT of extracted grid' )

  if resultImage is not None:
    baseX = (horizSpacing + maxWidth) * 2 + horizSpacing
    baseY = (vertSpacing  + maxHeight) * (rows) + 8
    drawText( baseX, baseY, 'result' )

  # Draw mode information

  str = 'show %s' % ('magnitudes' if showMagnitude else 'phases')
  glColor3f( 0.5, 0.2, 0.4 )
  drawText( windowWidth-len(str)*8-8, 12, str )

  # Done

  glfw.swap_buffers( window )

  

# Get information about how to place the images.
#
# toDraw                       2D array of complex images 
# rows, cols                   rows and columns in array
# maxHeight, maxWidth          max height and width of images
# scale                        amount by which to scale images
# horizSpacing, vertSpacing    spacing between images


def getImagesInfo():

  toDraw = [ [ image,   gridImage,   resultImage   ],
             [ image_FT, gridImage_FT, None ] ]

  rows = len(toDraw)
  cols = len(toDraw[0])

  # Find max image dimensions

  maxHeight = 0
  maxWidth  = 0
  
  for row in toDraw:
    for img in row:
      if img is not None:
        if img.shape[0] > maxHeight:
          maxHeight = img.shape[0]
        if img.shape[1] > maxWidth:
          maxWidth = img.shape[1]

  # Scale everything to fit in the window

  minSpacing = 30 # minimum spacing between images

  scaleX = (windowWidth  - (cols+1)*minSpacing) / float(maxWidth  * cols)
  scaleY = (windowHeight - (rows+1)*minSpacing) / float(maxHeight * rows)

  if scaleX < scaleY:
    scale = scaleX
  else:
    scale = scaleY

  maxWidth  = scale * maxWidth
  maxHeight = scale * maxHeight

  # Draw each image

  horizSpacing = (windowWidth-cols*maxWidth)/(cols+1)
  vertSpacing  = (windowHeight-rows*maxHeight)/(rows+1)

  return toDraw, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing
  

  
# Equalize the image histogram

def histoEq( pixels ):

  # build histogram

  h = [0] * 256 # counts

  width  = pixels.shape[0]
  height = pixels.shape[1]

  min = pixels.min()
  max = pixels.max()
  if max == min:
    max = min+1

  for i in range(width):
    for j in range(height):
      y = int( (pixels[i,j] - min) / (max-min) * 255 )
      h[y] = h[y] + 1

  # Build T[r] = s

  k = 256.0 / float(width * height) # common factor applied to all entries

  T = [0] * 256 # lookup table
  
  sum = 0
  for i in range(256):
    sum = sum + h[i]
    T[i] = int( math.floor(k * sum) - 1 )
    if T[i] < 0:
      T[i] = 0

  # Apply T[r]

  result = np.empty( pixels.shape )

  for i in range(width):
    for j in range(height):
      y = int( (pixels[i,j] - min) / (max - min) * 255 )
      result[i,j] = T[y]

  return result
  

# Handle keyboard input

def keyCallback( window, key, scancode, action, mods ):

  global image, image_FT, gridImage, gridImage_FT, resultImage, showMagnitude, doHistoEq, imageFilename, zoom, translate

  if action == glfw.PRESS:
    
    if key == glfw.KEY_ESCAPE:  # quit upon ESC
      sys.exit(0)

    elif key == glfw.KEY_L: # load an image

      if haveTK:
          imagePath = tkFileDialog.askopenfilename( initialdir = imageDir )
          if imagePath:
            image = loadImage( imagePath )
            imageFilename = os.path.basename( imagePath )
            image_FT = None
            gridImage = None
            gridImage_FT = None
            resultImage = None

    elif key == glfw.KEY_M: # toggle magnitude/phase display
      showMagnitude = not showMagnitude

    elif key == glfw.KEY_H: # toggle histogram equalization
      doHistoEq = not doHistoEq

    elif key == glfw.KEY_Z: # zero the zoom and translation
      zoom = 1
      translate = (0,0)

    elif key == glfw.KEY_C: # compute

      resultImage, lines = compute()
      print( 'Grid lines:' )
      for line in lines:
        print( '  angle %.1f, distance %.1f' % (line[0],line[1]) )

    elif key == glfw.KEY_DOWN:  # forward FT
      forwardFT_all()

    elif key == glfw.KEY_UP: # backward FT
      inverseFT_all()

    elif key == glfw.KEY_SLASH: # help (/ or ?)

      print( '''keys:
             c  compute the solution
             m  toggle between magnitude and phase in the FT  
             h  toggle histogram equalization in the FT  
             l  load image
    down arrow  forward transform
      up arrow  inverse transform
                translate with left mouse dragging
                zoom with right mouse draggin up/down
             z  reset the translation and zoom''' )



# Do a forward FT to image


def forwardFT_all():

  global image, image_FT

  if image is not None:
    image_FT = forwardFT( image )



# Do an inverse FT to image_FT


def inverseFT_all():

  global image, image_FT

  if image_FT is not None: 
    image = inverseFT( image_FT )


    
# Load an image
#
# Return the image as a 2D numpy array of complex_ values.


def loadImage( path ):

  try:
    img = Image.open( path ).convert( 'L' ).transpose( Image.FLIP_TOP_BOTTOM )
  except:
    print( 'Failed to load image %s' % path )
    sys.exit(1)

  img = ImageOps.invert(img)

  return np.array( list( img.getdata() ), np.complex_ ).reshape( (img.size[1],img.size[0]) )



# Handle window reshape

def reshape( newWidth, newHeight ):

  global windowWidth, windowHeight

  windowWidth  = newWidth
  windowHeight = newHeight

  glViewport( 0, 0, windowWidth, windowHeight )



# Output an image
#
# The image has complex values, so output either the magnitudes or the
# phases, according to the 'outputMagnitudes' parameter.

def outputImage( image, filename, outputMagnitudes, isFT, invert ):

  if not isFT:
    show = np.real(image)
  else:
    ak =  2 * np.real(image)
    bk = -2 * np.imag(image)
    if outputMagnitudes:
      show = np.log( 1 + np.sqrt( ak*ak + bk*bk ) ) # take the log because there are a few very large values (e.g. the DC component)
    else:
      show = np.arctan2( -1 * bk, ak )
    show = np.fft.fftshift( show ) # shift FT so that origin is in centre

  min = show.min()
  max = show.max()

  img = Image.fromarray( np.uint8( (show - min) * (255 / (max-min)) ) ).transpose( Image.FLIP_TOP_BOTTOM )

  if invert:
    img = ImageOps.invert(img) 

  img.save( filename )




# Draw text in window

def drawText( x, y, text ):

  glRasterPos( x, y )
  for ch in text:
    glutBitmapCharacter( GLUT_BITMAP_8_BY_13, ord(ch) )

    

# Handle window reshape


def windowReshapeCallback( window, newWidth, newHeight ):

    global windowWidth, windowHeight

    windowWidth  = newWidth
    windowHeight = newHeight


# Handle mouse click


currentButton = None
initX = 0
initY = 0
initZoom = 0
initTranslate = (0,0)

def mouseButtonCallback( window, btn, action, keyModifiers ):

  global currentButton, initX, initY, initZoom, initTranslate, translate, zoom

  x,y = glfw.get_cursor_pos( window )

  if action == glfw.PRESS:

    currentButton = btn
    initX = x
    initY = y
    initZoom = zoom
    initTranslate = translate

  elif action == glfw.RELEASE:

    currentButton = None

    if btn == glfw.MOUSE_BUTTON_LEFT and initX == x and initY == y: # Process a left click (with no dragging)

      # Find which image the click is in

      toDraw, rows, cols, maxHeight, maxWidth, scale, horizSpacing, vertSpacing = getImagesInfo()

      row = (y-vertSpacing ) / float(maxHeight+vertSpacing)
      col = (x-horizSpacing) / float(maxWidth+horizSpacing)

      if row < 0 or row-math.floor(row) > maxHeight/(maxHeight+vertSpacing):
        return

      if col < 0 or col-math.floor(col) > maxWidth/(maxWidth+horizSpacing):
        return

      # Get the image

      image = toDraw[ int(row) ][ int(col) ]

      if image is None:
        return

      # Get bounds of visible image
      #
      # Bounds are [cx-offset,cx+offset] x [cy-offset,cy+offset]
      
      height = scale * image.shape[0]
      width  = scale * image.shape[1]

      cx     = 0.5 - translate[0]/width
      cy     = 0.5 - translate[1]/height
      offset = 0.5 / zoom

      # Find pixel position within the image array

      xFraction = (col-math.floor(col)) / (maxWidth /float(maxWidth +horizSpacing))
      yFraction = (row-math.floor(row)) / (maxHeight/float(maxHeight+vertSpacing ))

      pixelX = int( image.shape[1] * ((1-xFraction)*(cx-offset) + xFraction*(cx+offset)) )
      pixelY = int( image.shape[0] * ((1-yFraction)*(cy+offset) + yFraction*(cy-offset)) )
      
      # for the FT images, move the position half up and half right,
      # since the image is displayed with that shift, while the FT array
      # stores the unshifted values.

      isFT = (int(row) == 1)

      if isFT:

        pixelX = pixelX - image.shape[1]/2
        if pixelX < 0:
          pixelX = pixelX + image.shape[1]

        pixelY = pixelY - image.shape[0]/2
        if pixelY < 0:
          pixelY = pixelY + image.shape[0]

      # Perform the operation
      #
      # No operation is implemented here, but could be (e.g. image modification at the mouse position)

      # applyOperation( image, pixelX, pixelY, isFT )  

      print( 'click at', pixelX, pixelY, '=', image[pixelY][pixelX], np.absolute(image[pixelY][pixelX]) )

      # Done



# Handle mouse motion.  We don't want to transform the image and
# redraw with each tiny mouse movement.  Instead, just record the fact
# that the mouse moved.  After events are process in
# glfw.wait_events(), check whether the mouse moved and, if so, act on
# it.

mousePositionChanged = False

def mouseMovementCallback( window, x, y ):

  global mousePositionChanged

  if currentButton is not None: # button is down
    mousePositionChanged = True



# Handle mouse dragging
#
# Zoom out/in with right button dragging up/down.
# Translate with left button dragging.


def actOnMouseMovement( window, x, y ):

  global zoom, translate

  if currentButton == glfw.MOUSE_BUTTON_RIGHT:

    # zoom

    factor = 1 # controls the zoom rate
    
    if y > initY: # zoom in
      zoom = initZoom * (1 + factor*(y-initY)/float(windowHeight))
    else: # zoom out
      zoom = initZoom / (1 + factor*(initY-y)/float(windowHeight))

  elif currentButton == glfw.MOUSE_BUTTON_LEFT:

    # translate

    translate = ( initTranslate[0] + (x-initX)/zoom, initTranslate[1] + (initY-y)/zoom )


# For an image coordinate, if it's < 0 or >= max, wrap the coorindate
# around so that it's in the range [0,max-1].  This is useful dealing
# with FT images.

def wrap( val, max ):

  if val < 0:
    return val+max
  elif val >= max:
    return val-max
  else:
    return val



# Initialize GLFW and run the main event loop

def main_interactive():

    if not glfw.init():
        print( 'Error: GLFW failed to initialize' )
        sys.exit(1)

    window = glfw.create_window( windowWidth, windowHeight, "Assignment 2", None, None )

    if not window:
        glfw.terminate()
        print( 'Error: GLFW failed to create a window' )
        sys.exit(1)

    glfw.make_context_current( window )

    glfw.swap_interval( 1 )  # redraw at most every 1 screen scan

    # Use GLUT for bitmapped characters

    glutInit()

    # Callbacks
    
    glfw.set_key_callback( window, keyCallback )
    glfw.set_window_size_callback( window, windowReshapeCallback )
    glfw.set_mouse_button_callback( window, mouseButtonCallback )
    glfw.set_cursor_pos_callback( window, mouseMovementCallback )

    loadImage( os.path.join( imageDir, imageFilename ) )

    display( window )

    # Main event loop

    prevX, prevY = glfw.get_cursor_pos( window )

    while not glfw.window_should_close( window ):

        glfw.wait_events()

        currentX, currentY = glfw.get_cursor_pos( window )
        if currentX != prevX or currentY != prevY:
          actOnMouseMovement( window, currentX, currentY )

        display( window )

    glfw.destroy_window( window )
    glfw.terminate()



# Load initial data
#
# The command line (stored in sys.argv) could have:
#
#     main.py {image filename}

if len(sys.argv) > 1:
  imageFilename = sys.argv[1]
  imagePath = os.path.join( imageDir,  imageFilename  )

image  = loadImage( imagePath )


# If commands exist on the command line (i.e. there are more than two
# arguments), process each command, then exit.  Otherwise, go into
# interactive mode.
#
# DO NOT MODIFY THIS CODE, AS IT IS USED FOR TESTING THE ASSIGNMENT.
#
# You can use this for your own testing, if you wish.

if len(sys.argv) <= 2:

  main_interactive()

else:

  # process commands
      
  outputMagnitudes = True

  # process commands

  cmds = sys.argv[2:]

  while len(cmds) > 0:
    cmd = cmds.pop(0)
    if cmd == 'f':
      forwardFT_all()
    elif cmd == 'i':
      inverseFT_all()
    elif cmd == 'm':
      outputMagnitudes = True
    elif cmd == 'p':
      outputMagnitudes = False
    elif cmd == 'c':
      image, lines = compute()
      print( lines )
    elif cmd[0] == 'o': # image name follows in 'cmds'
      filename = cmds.pop(0)
      outputImage( resultImage, filename, False, False, True )
    else:
      print( """command '%s' not understood.
command-line arguments:
  c - compute  
  f - apply forward FT
  i - apply inverse FT
  o - output the image
  m - for output, use magnitudes (default)
  p - for output, use phases""" % cmd )
