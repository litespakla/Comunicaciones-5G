import sys
import wave
from PIL import Image

def codificador_fuente(filename):

    #Identificar el tipo de archivo
    match filename[-3:]:

        #Archivo de texto
        case 'txt':
            with open(filename, 'r') as file:
                data = file.read()

        #Archivo de audio
        case 'wav':
            with wave.open(filename, 'rb') as wav_file:
                data = wav_file.readframes(wav_file.getnframes())

                #Información de los parámetros de audio
                params=wav_file.getparams()

                #Codificar los parámetros
                encoded_params = ''
                text=False
                for param in params:

                    #Para los parámetros que son números
                    if isinstance(param, int):
                        encoded_params += format(param, '032b')

                    #Para los paámetros que son texto
                    elif isinstance(param, str):

                        #Para el primer parámetro que no es texto
                        if not text:

                            #Longitud de los bits que representan números
                            lenght_params=format(len(encoded_params), '032b')
                            text=True

                        #Ya se indicó la longitud de los bits que son números
                        else:
                            lenght_params=''

                        #Codificación del string
                        encoded_params += format(8*len(param), '032b')
                        encoded_params += ''.join(format(ord(char), '08b') for char in param)

                        #Longitud del string
                        encoded_params=lenght_params+ encoded_params 

                #Longitud de la codificación de los parámetros
                lenght_params=format(32+len(encoded_params), '032b')
                encoded_params=lenght_params+ encoded_params

        #Imagen
        case 'bmp':
            img = Image.open(filename)
            data=[]

            #Dimensiones de la imagen
            width, height = img.size

            #Recorrer la imagen pixel por pixel para recuperar la información RGB
            for y in range(height):
                for x in range(width):
                    data.append(img.getpixel((x, y)))

    #Lista de caracteres
    vT = []  

    #Arreglo bkT
    bkT=[]

    #Los primeros dos bytes son las dimensiones de la imagen
    if filename[-3:]=='bmp':
        bkT.append(format(height, '016b'))
        bkT.append(format(width, '016b'))

    #Los primeros bits son los parámetros del audio
    elif filename[-3:]=='wav':
        bkT.append(encoded_params)

    #Ciclo
    for vk in data:

        #Almacenar vk en VT
        vT.append(vk)

        #Convertir el símbolo vk a su representación binaria de 8 bits
        match filename[-3:]:

            #Archivo de texto
            case 'txt':
                bkl = format(ord(vk), '08b')

            #Archivo de audio
            case 'wav':
                bkl = format(vk, '016b')

            #Imagen
            case 'bmp':
                bkl=''
                for rgb in vk:
                    bkl+=format(rgb, '08b')
        
        #Almacenar la secuencia de bits
        bkT.append(bkl) 

    #Concatenar secuencias de bits
    bfT = ''.join(bkT)
    return bfT

def descodificador_fuente(bfR, filename):

    #Dividir la secuencia de bits en bloques de 8 bits
    bkR = [bfR[i:i+8] for i in range(0, len(bfR), 8)]  
        
    #Reconstruir el archivo
    match filename[-3:]:

        #Archivo de texto
        case 'txt':

            #Crear la salida
            with open(filename, 'w') as file:
                file.write('')  

            #Ciclo
            for binary in bkR:

                #Convertir secuencias de bits a caracteres
                vR = chr(int(binary, 2))  

                #Escribir caracteres en el archivo de salida
                with open(filename, 'a') as file:
                    file.write(''.join(vR))  

        #Archivo de audio
        case 'wav':

            #Longitud de los parámetros (primeros 32 bits)
            lenght=int(bfR[0:32], 2)

            #Longitud de los parámetros que son números (segundos 32 bits)
            lenght_int=int(bfR[32:64], 2)
            
            #Decodificar los parámetros de audio (int)
            decoded_params = []
            for i in range(64, 64+lenght_int, 32):
                decoded_params.append(int(bfR[i:i+32], 2))

            #Decodificar los parámetros de audio (str)
            pos=64+lenght_int
            while pos<lenght:

                #Longitud de la palabra
                word_lenght=int(bfR[pos:32+pos], 2)

                #Decodificar la palabra
                decoded_params.append(''.join(chr(int(bfR[i:i+8], 2)) for i in range(32+pos, 32+pos+word_lenght, 8)))

                #Mover la posición
                pos=32+pos+word_lenght

            #Parámetros recuperados del audio original
            params = tuple(decoded_params)

            #Ciclo
            audio_data = bytes([int(bkR[k]+bkR[k+1], 2) for k in range(int(lenght/8), len(bkR), 2)])

            # Escribir los bytes en un nuevo archivo .wav
            with wave.open(filename, 'wb') as audio_file:
                audio_file.setparams(params)
                audio_file.writeframes(audio_data)

        #Imagen
        case 'bmp':
            
            #Largo y ancho
            height=int(bkR[0]+bkR[1], 2)
            width=int(bkR[2]+bkR[3], 2)

            #Crear una nueva imagen con la misma resolución
            img = Image.new('RGB', (width, height))

            #Ciclo
            k=4
            for y in range(height):
                for x in range(width):

                    #Configurar los valores de los píxeles
                    r=int(bkR[k], 2)
                    g=int(bkR[k+1], 2)
                    b=int(bkR[k+2], 2)
                    img.putpixel((x, y), (r, g, b))
                    k+=3

            #Guardar la imagen reconstruida en el archivo de salida
            img.save(filename)

#Main
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    #Codificar el archivo
    bfT = codificador_fuente(input_file)

    #Canal ideal. Los bits recibidos son iguales a la información
    bfR=bfT

    #Decodificar la información
    descodificador_fuente(bfR, output_file)
