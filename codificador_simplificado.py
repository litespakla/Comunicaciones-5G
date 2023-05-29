import sys
import wave
import numpy as np
import random
from PIL import Image

#Bits de información
k=4

#Bits de la palabra codificada
n=7

#Matriz P
P= np.array([[1,1,0], [0,1,1], [1,1,1], [1,0,1]])

#Matriz generadora
G=np.hstack((P, np.eye(k)))

#Matriz de paridad
H=np.vstack((np.eye(n-k), P))

#Convierte el archivo en binario
def codificador_binario(filename):

    #Identificar el tipo de archivo
    match filename[-3:]:

        #Archivo de texto
        case 'txt':
            with open(filename, 'r') as file:
                data = file.read()
                params=''

        #Archivo de audio
        case 'wav':
            with wave.open(filename, 'rb') as wav_file:
                data = wav_file.readframes(wav_file.getnframes())

                #Información de los parámetros de audio
                params=wav_file.getparams()

        #Imagen
        case 'bmp':
            img = Image.open(filename)
            data=[]

            #Dimensiones de la imagen
            width, height = img.size
            params=[width, height]

            #Recorrer la imagen pixel por pixel para recuperar la información RGB
            for y in range(height):
                for x in range(width):
                    data.append(img.getpixel((x, y)))

    #Lista de caracteres
    vT = []  

    #Arreglo bkT
    bkT=[]

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
                bkl = format(vk, '08b')

            #Imagen
            case 'bmp':
                bkl=''
                for rgb in vk:
                    bkl+=format(rgb, '08b')
        
        #Almacenar la secuencia de bits
        bkT.append(bkl) 

    #Concatenar secuencias de bits
    bfT = ''.join(bkT)

    return bfT, params

#A partir del binario se reconstruye el archivo
def decodificador_binario(bfR, filename, params):

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

            #Ciclo
            audio_data = bytes([int(bkR[k], 2) for k in range(0, len(bkR))])

            # Escribir los bytes en un nuevo archivo .wav
            with wave.open(filename, 'wb') as audio_file:
                audio_file.setparams(params)
                audio_file.writeframes(audio_data)

        #Imagen
        case 'bmp':
            
            #Largo y ancho
            height=params[1]
            width=params[0]

            #Crear una nueva imagen con la misma resolución
            img = Image.new('RGB', (width, height))

            #Ciclo
            j=0
            for y in range(height):
                for x in range(width):

                    #Configurar los valores de los píxeles
                    r=int(bkR[j], 2)
                    g=int(bkR[j+1], 2)
                    b=int(bkR[j+2], 2)
                    img.putpixel((x, y), (r, g, b))
                    j+=3

            #Guardar la imagen reconstruida en el archivo de salida
            img.save(filename)

#Codifica la información con la matriz G
def codificador_canal(bfT):

    #Verificar que la secuencia sea de k bits
    ceros=0
    if len(bfT)%k!=0:
        ceros=k-(len(bfT)%k)

    #Los primeros k bits van a ser la cantidad de ceros agregada
    bfT = format(ceros, '0' + str(k) +'b') + '0'*ceros + bfT

    #Dividir la secuencia de bits en bloques de k bits
    bfl = [bfT[i:i+k] for i in range(0, len(bfT), k)] 

    #Bits transmitidos
    bcl=[]

    #Codificar cada bloque
    for block in bfl:

        #Multiplicar uG
        m=[int(char) for char in block]
        u=np.dot(m, G)

        #Módulo 2
        u=''.join([str(int(i%2)) for i in u])

        #Almacenar la secuencia codificadas
        bcl.append(u)

    #Secuencia transmitida
    bcT=''.join(bcl)

    return bcT

#Decodifica la información con la matriz H
def decodificador_canal(bcR):

    #Dividir la secuencia de bits en bloques de n bits
    bcl = [bcR[i:i+n] for i in range(0, len(bcR), n)] 

    #Bits recibidos
    bfl=[]

    #Decodificar cada bloque
    for block in bcl:

        #Sindrome
        v=[int(char) for char in block]
        S=np.dot(v, H)
        S=''.join([str(int(i%2)) for i in S])

        #Comparar el síndrome con H
        for i in range(len(H)):
            file=''.join([str(int(j)) for j in H[i]])

            #Si el síndrome es igual a alguna de las filas de H
            if file==S:

                #Corregir el error
                if block[i]=='0':
                    block = block[:i] + '1' + block[i+1:]
                else:
                    block = block[:i] + '0' + block[i+1:]
                break

        #Recuperar la información recibida (últimos k bits del bloque)
        m=block[n-k:]
        bfl.append(m)

    #Secuencia recibida
    bfR=''.join(bfl)

    #Primeros k bits son la cantidad de ceros agregada
    ceros = int(bfR[0:k], 2)

    #Se elimima la cantidad agregada de ceros
    bfR=bfR[k+ceros:]

    return bfR

#Introduce errores en la secuencia
def error(sequence, error):

    # Convertir la secuencia a una lista para hacerla mutable
    sequence_list = list(sequence)
    length = len(sequence_list)

    # Calcular el número de errores
    num_errors = int(length * error)

    # Generar índices de errores
    error_indices = random.sample(range(length), num_errors)

    # Agregar errores a la secuencia
    for i in error_indices:
        sequence_list[i] = '1' if sequence_list[i] == '0' else '0'
    
    # Convertir la lista de secuencia de nuevo a una cadena
    sequence = ''.join(sequence_list)

    return sequence

#Prueba de canal ideal
def prueba_canal_ideal():
    print('Prueba del canal ideal\n')

    # Longitud de la secuencia (debe ser par para tener la misma cantidad de 1s y 0s)
    length = 74

    # Genera la secuencia con la misma cantidad de 1s y 0s
    sequence = np.array([0]*(length//2) + [1]*(length//2))

    # Mezcla la secuencia para que los bits estén en orden aleatorio
    np.random.shuffle(sequence)

    #Secuencia de bits de información
    bfl=''.join([str(i) for i in sequence])
    print('Información: ', bfl, '\n')

    #Secuencia de bits transmitidos
    bcl=codificador_canal(bfl)
    print('Bits transmitidos: ', bcl, '\n')

    #Canal ideal. Los bits recibidos son iguales a los bits transmitidos
    bcl_R=bcl

    #Bits de información recibidos
    bfl_R=decodificador_canal(bcl_R)
    print('Información recibida: ', bfl_R, '\n')

    #Verificar que no hubo errores de transmisión
    if bfl==bfl_R:
        print('No hubo errores de transmisión')

#Prueba de canal binario simétrico
def prueba_canal_binario():
    print('Prueba del canal binario simétrico\n')

    # Longitud de la secuencia (debe ser par para tener la misma cantidad de 1s y 0s)
    length = 250

    # Genera la secuencia con la misma cantidad de 1s y 0s
    sequence = np.array([0]*(length//2) + [1]*(length//2))

    # Mezcla la secuencia para que los bits estén en orden aleatorio
    np.random.shuffle(sequence)

    #Secuencia de bits de información
    bfl=''.join([str(i) for i in sequence])
    print('Información: ', bfl, '\n')

    #Secuencia de bits transmitidos
    bcl=codificador_canal(bfl)
    print('Secuencia transmitida: ', bcl, '\n')

    #Error
    e=0.05

    #Canal binario simétrico. Se introduce un error p(e)
    bcl_R=error(bcl, e)
    print('Secuencia recibida: ', bcl_R, '\n')

    #Contar la cantidad de errores en la secuencia recibida
    fails=0
    for i in range(len(bcl)):
        if bcl[i]!=bcl_R[i]:
            fails+=1
    print('Hubo ', fails, ' fallos en la secuencia de bits recibida. La razón de error es ', fails/len(bcl), '\n')

    #Bits de información recibidos
    bfl_R=decodificador_canal(bcl_R)
    print('Información recibida: ', bfl_R, '\n')

    #Contar la cantidad de errores en la información recibida
    fails=0
    for i in range(len(bfl)):
        if bfl[i]!=bfl_R[i]:
            fails+=1
    print('Hubo ', fails, ' fallos en los bits de información recibidos. La razón de error es ', fails/len(bfl))

#Main
if __name__ == '__main__':
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    #Pruebas
    prueba_canal_ideal()
    print('------------------------------------------------\n')
    prueba_canal_binario()
    print('------------------------------------------------\n')

    #Archivos
    print('Prueba para un archivo')

    #Agregar la extensión
    output_file= output_file + input_file[-4:]

    #Codificar el archivo
    bfT, params = codificador_binario(input_file)

    #Información enviada a través del canal
    bcR=codificador_canal(bfT)

    #Error
    e=0.05

    #Canal binario simétrico. Se introduce un error p(e)
    bcl=error(bcR, e)

    #Contar la cantidad de errores en la secuencia recibida
    fails=0
    for i in range(len(bcl)):
        if bcl[i]!=bcR[i]:
            fails+=1
    print('Hubo ', fails, ' fallos en la secuencia de bits recibida. La razón de error es ', fails/len(bcl), '\n')

    #Bits de información recibidos
    bfR=decodificador_canal(bcl)

    #Contar la cantidad de errores en la información recibida
    fails=0
    for i in range(len(bfR)):
        if bfR[i]!=bfT[i]:
            fails+=1
    print('Hubo ', fails, ' fallos en los bits de información recibidos. La razón de error es ', fails/len(bfT))
    
    #Decodificar la información
    decodificador_binario(bfR, output_file, params)