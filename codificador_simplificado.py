import sys
import wave
import numpy as np
import random
import matplotlib.pyplot as plt
from PIL import Image
from scipy import signal

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

#Número de símbolos en la modulación PAM y ASK
M = 4

#Número de bits por símbolo
b = int(np.log2(M)) 

#Número de muestras por símbolo
Ns = 20 

#Relación señal a ruido
SNR_dB = 15

#Frecuencia de la portadora (en Hz) ASK
fc = 1000

#Periodo de la portadora (en segundos)
Ts = 1 / fc  

#Frecuencia de muestreo en Hz
Fs = int(Ns/Ts)   

#Amplitud de la señal
A = 1          

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
    #print(bcl)
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

#Modular la señal en PAM
def generar_pulso_PAM(bcl):

    #Verificar que la secuencia sea de b bits
    ceros=0
    if len(bcl)%b!=0:
        ceros=b-(len(bcl)%b)

    #Los primeros b bits van a ser la cantidad de ceros agregada
    bcl = format(ceros, '0' + str(b) +'b') + '0'*ceros + bcl

    #Pasar la secuencia a un arreglo de bits
    bcl=np.array([int(i) for i in bcl])

    # Definición de asignación de bits a símbolos
    asignacion = np.linspace(-1, 1, 2**b)

    # Mapeo de bits a símbolos
    bloques = bcl.reshape(-1, b)
    simbolos_modulados = np.array([asignacion[int(np.dot(
        bloque, [2**(b-i-1) for i in range(b)]))] for bloque in bloques]).flatten()

    # Generación del pulso p(t)
    p = np.ones(Ns)

    # Generación de la señal modulada x(k)
    xT = np.repeat(simbolos_modulados, Ns)

    return xT

#Agregar ruido a la señal
def simular_medio_transmision_ruidoso(xT):

    # Generación del ruido
    N = np.random.normal(0, 1, len(xT))
    
    # Calcular la potencia de la señal transmitida
    Pt = np.sum(np.abs(xT)**2) / len(xT)
    
    # Calcular la potencia del ruido según la relación SNR deseada
    SNR = 10**(SNR_dB / 10)
    Pr = Pt / SNR
    
    # Añadir el ruido a la señal transmitida
    xR = xT + np.sqrt(Pr) * N
    
    return xR

#Desmodular señal PAM
def desmodulador_PAM(xR):

    # Definición de asignación de símbolos a bits
    bits_a_simbolos = {format(i, '0'+str(b)+'b'): (2*i/(M-1))-1 for i in range(M)}
    simbolos_a_bits = {v: k for k, v in bits_a_simbolos.items()}

    # Dividir la secuencia de muestras en bloques de longitud Ns
    xR = xR.reshape(-1, Ns)

    # Calculamos el promedio de cada segmento
    y = np.mean(xR, axis=1)

    # Decidimos qué símbolo representa mejor cada promedio
    a_estrella = [min(bits_a_simbolos.values(), key=lambda x:abs(x-y[i])) for i in range(len(y))]

    # Convertimos los símbolos a bits
    bcR = [simbolos_a_bits[a_estrella[i]] for i in range(len(a_estrella))]

    # Convertimos la lista de cadenas de bits en un arreglo de bits
    bcR = np.array([list(map(int, bcR[i])) for i in range(len(bcR))]).flatten()

    #Primeros b bits son la cantidad de ceros agregada
    ceros = int(''.join(str(s) for s in bcR[0:b]), 2)

    #Se elimima la cantidad agregada de ceros
    bcR=bcR[b+ceros:]

    return np.array(bcR)

#Modular la señal en ASK
def generar_pulso_ASK(bcl):

    #Verificar que la secuencia sea de b bits
    ceros=0
    if len(bcl)%b!=0:
        ceros=b-(len(bcl)%b)

    #Los primeros b bits van a ser la cantidad de ceros agregada
    bcl = format(ceros, '0' + str(b) +'b') + '0'*ceros + bcl

    #Pasar la secuencia a un arreglo de bits
    bcl=np.array([int(i) for i in bcl])

    #Definir asignación de bits a símbolos
    symbol_mapping = {i: (2 * i / (M - 1)) for i in range(M)}

    #Inicializar arreglo para almacenar la secuencia de símbolos
    symbols = np.zeros(len(bcl) // b, dtype=int)

    #Asignar bits a símbolos
    for i in range(len(bcl) // b):
        bits = bcl[i * b:(i + 1) * b]
        symbol = 0
        for j in range(b):
            symbol += bits[j] * 2**(b - j - 1)
        symbols[i] = symbol

    #Generar la señal portadora c(t)
    t = np.linspace(0, Ts, Ns)
    c = np.cos(2 * np.pi * fc * t)

    #Generar la señal modulada x(t)
    s = np.zeros(len(bcl) * Ns // b)
    for i, symbol in enumerate(symbols):
        s[i * Ns:(i + 1) * Ns] = symbol_mapping[symbol] * c

    return s

#Desmodular señal ASK
def desmodulador_ASK(sR):
        
    # Definición de asignación de símbolos a bits
    bits_a_simbolos = {format(i, '0'+str(b)+'b'): (2*i/(M-1)) for i in range(M)}
    simbolos_a_bits = {v: k for k, v in bits_a_simbolos.items()}

    # Dividir la secuencia de muestras en bloques de longitud Ns
    sR = sR.reshape(-1, Ns)

    #Detectar la envolvente de cada secuencia de muestras
    envelope = np.abs(signal.hilbert(sR))  

    #Muestrear la envolvente a la mitad de cada período de la portadora
    yn = envelope[:, Ns//2 - 1]  

    # Decidimos qué símbolo representa mejor cada promedio
    a_estrella = [min(bits_a_simbolos.values(), key=lambda x:abs(x-yn[i])) for i in range(len(yn))]

    # Convertimos los símbolos a bits
    bcR = [simbolos_a_bits[a_estrella[i]] for i in range(len(a_estrella))]

    # Convertimos la lista de cadenas de bits en un arreglo de bits
    bcR = np.array([list(map(int, bcR[i])) for i in range(len(bcR))]).flatten()

    #Primeros b bits son la cantidad de ceros agregada
    ceros = int(''.join(str(s) for s in bcR[0:b]), 2)

    #Se elimima la cantidad agregada de ceros
    bcR=bcR[b+ceros:]

    return np.array(bcR)

#Main
if __name__ == '__main__':
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    #Archivos
    print('Prueba para un archivo')

    #Agregar la extensión
    output_file1= output_file + '_canal_simetrico_PAM' + input_file[-4:]
    output_file2= output_file + '_canal_ideal_PAM' + input_file[-4:]
    output_file3= output_file + '_canal_simetrico_ASK' + input_file[-4:]
    output_file4= output_file + '_canal_ideal_ASK' + input_file[-4:]

    #Codificar el archivo (bits de información a la salida del codificador de la fuente)
    bfT, params = codificador_binario(input_file)

    #Información transmitida (a la salida del codificador de canal)
    bcR=codificador_canal(bfT)

    #Error
    e=0.02

    #Canal binario simétrico. Se introduce un error p(e)
    bcl_b=error(bcR, e)

    #Canal ideal 
    bcl_i=bcR
    
    #Modular para obtener la secuencia de muestras
    xT_b = generar_pulso_PAM(bcl_b) #Simétrico PAM
    xT_i = generar_pulso_PAM(bcl_i) #Ideal PAN
    sT_b = generar_pulso_ASK(bcl_b) #Simétrico ASK
    sT_i = generar_pulso_ASK(bcl_i) #Ideal ASK

    #Medio de transmisión con ruido
    xR_b = simular_medio_transmision_ruidoso(xT_b) #Simétrico PAM
    xR_i = simular_medio_transmision_ruidoso(xT_i) #Ideal PAM
    sR_b = simular_medio_transmision_ruidoso(sT_b) #Simétrico ASK
    sR_i = simular_medio_transmision_ruidoso(sT_i) #Ideal  ASK

    # Desmodular la señal recibida y obtener los bits recibidos
    bcl_b = desmodulador_PAM(xR_b)  #Simétrico PAM
    bcl_b=''.join(str(s) for s in bcl_b)
    bcl_i = desmodulador_PAM(xR_i)  #Ideal PAM
    bcl_i=''.join(str(s) for s in bcl_i)
    bcl_bA = desmodulador_ASK(sR_b)  #Simétrico ASK
    bcl_bA=''.join(str(s) for s in bcl_bA)
    bcl_iA = desmodulador_ASK(sR_i)  #Ideal ASK
    bcl_iA=''.join(str(s) for s in bcl_iA)

    #Contar la cantidad de errores en la secuencia recibida
    fails_b=0
    fails_i=0
    fails_bA=0
    fails_iA=0
    for i in range(len(bcl_b)):
        if bcl_b[i]!=bcR[i]:
            fails_b+=1
        if bcl_i[i]!=bcR[i]:
            fails_i+=1
    for i in range(len(bcl_bA)):
        #sprint(len(bcR), len(bcl_bA))
        if bcl_bA[i]!=bcR[i]:
            fails_bA+=1
        if bcl_iA[i]!=bcR[i]:
            fails_iA+=1
    print('Hubo ', fails_b, ' fallos en la secuencia de bits recibida para el canal binario simétrico con modulación PAM. La razón de error es ', fails_b/len(bcl_b))
    print('Hubo ', fails_i, ' fallos en la secuencia de bits recibida para el canal ideal con modulación PAM. La razón de error es ', fails_i/len(bcl_i), '\n')
    print('Hubo ', fails_bA, ' fallos en la secuencia de bits recibida para el canal binario simétrico con modulación ASK. La razón de error es ', fails_bA/len(bcl_bA))
    print('Hubo ', fails_iA, ' fallos en la secuencia de bits recibida para el canal ideal con modulación ASK. La razón de error es ', fails_iA/len(bcl_iA), '\n')

    #Bits de información recibidos
    bfR_b=decodificador_canal(bcl_b)
    bfR_i=decodificador_canal(bcl_i)
    bfR_bA=decodificador_canal(bcl_bA)
    bfR_iA=decodificador_canal(bcl_iA)

    #Contar la cantidad de errores en la información recibida
    fails_b=0
    fails_i=0
    fails_bA=0
    fails_iA=0
    for i in range(len(bfR_b)):
        if bfR_b[i]!=bfT[i]:
            fails_b+=1
        if bfR_i[i]!=bfT[i]:
            fails_i+=1
    for i in range(len(bfR_bA)):
        if bfR_bA[i]!=bfT[i]:
            fails_bA+=1
        if bfR_iA[i]!=bfT[i]:
            fails_iA+=1
    print('Hubo ', fails_b, ' fallos en los bits de información recibidos para el canal binario simétrico con modulación PAM. La razón de error es ', fails_b/len(bfT))
    print('Hubo ', fails_i, ' fallos en los bits de información recibidos para el canal ideal con modulación PAM. La razón de error es ', fails_i/len(bfT), '\n')
    print('Hubo ', fails_bA, ' fallos en los bits de información recibidos para el canal binario simétrico con modulación ASK. La razón de error es ', fails_bA/len(bfT))
    print('Hubo ', fails_iA, ' fallos en los bits de información recibidos para el canal ideal con modulación ASK. La razón de error es ', fails_iA/len(bfT))
    
    #Decodificar la información
    decodificador_binario(bfR_b, output_file1, params)
    decodificador_binario(bfR_i, output_file2, params)
    decodificador_binario(bfR_bA, output_file3, params)
    decodificador_binario(bfR_iA, output_file4, params)

    # Gráfico del pulso del PAM primeros 10 símbolos
    plt.plot(xT_b[0:10*Ns], color='black', linestyle='--', label='Pulso del PAM')
    # Gráfico del pulso con ruido GWN para PAM
    plt.plot(xR_b[0:10*Ns], label='Pulso con ruido GWN')
    plt.xlabel('Tiempo')
    plt.ylabel('Amplitud')
    plt.title('Comparación del pulso del PAM con y sin ruido')
    plt.legend()
    plt.show()    

    # Definir ejes de tiempo para la señal modulada y portadora
    t_mod = np.linspace(0, len(sR_i[0:10*Ns]) / Ns, len(sR_i[0:10*Ns]))
    c = np.cos(2 * np.pi * fc * t_mod)

    # Graficar la señal portadora
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 3, 1)
    plt.plot(t_mod, c)
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Señal Portadora')
    plt.grid(True)

    # Graficar la señal modulada ASK primeros 10 símbolos
    plt.subplot(1, 3, 2)
    plt.plot(t_mod, sR_i[0:10*Ns])
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Señal Modulada canal ideal')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    plt.plot(t_mod, sR_b[0:10*Ns])
    plt.xlabel('Tiempo (s)')
    plt.ylabel('Amplitud')
    plt.title('Señal Modulada canal binario simétrico')
    plt.grid(True)
    plt.show()
