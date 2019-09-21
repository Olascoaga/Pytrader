#!/usr/bin/python
# -*- coding: utf-8 -*-

from iqoptionapi.api import IQOptionAPI
from datetime import datetime
from datetime import time as tiempo
import time
import numpy as np
import threading

api = IQOptionAPI("iqoption.com", "usuario", "contraseña")
api.connect()

def conectar(id_activo):
    for i in range(4):
    	api = IQOptionAPI("iqoption.com", "usuario", "contraseña")
    	api.connect()
    for j in range(4):
    	api.getcandles(1,1)
    	time.sleep(0.5)
    	velas = api.candles.candles_data
    	time.sleep(30)
    return velas

def obtener_velas(id_activo):

    try:
        # print "Peticion de velas:", "-", datetime.now()
        api.getcandles(id_activo, 1)  # produce una excepción cuándo el websocket pierde conexión con el servidor
    except:
        print "¡Desconexión!", datetime.now()

    time.sleep(0.5)
    if api.candles.current_candle.candle_close != 0:
        cierre = api.candles.current_candle.candle_close / 1000000.0
    else:
        cierre = api.candles.candles_data[-2][2] / 1000000.0
    vela = [api.candles.first_candle.candle_time,
            api.candles.first_candle.candle_open / 1000000.0,
            cierre]
    # print "Hora apertura:", activo, datetime.fromtimestamp(api.candles.first_candle.candle_time)
    # print "Hora cierre:", datetime.fromtimestamp(api.candles.current_candle.candle_time)
    #print "Valor cierre:", cierre, "\n"
    

    return vela  # Para vela: [0] = fecha, [1] = Apertura, [2] = Cierre

# Constantes
inversion = 3 #Cambiar cada mes
hora_cero = True
velas = None
activo = 76
api.timesync.expiration_time = 1
senal = 0  # 0 = sin senal, 1 = alza, 2 = baja
# Configuración estrategia EURUSD
sma_eurusd = []
periodos_eurusd = 20
desviaciones_eurusd = 1
alta_eurusd = 0
baja_eurusd = 0
# Configuración de metas
meta = True

while True:
    if velas is not None:
        if hora_cero:
            print "Capital Inicial:", capital_inicial
            while datetime.now().second != 0:
                time.sleep(0.01)
                hora_cero = False

	while datetime.now().second % 10 != 0:
	    time.sleep(0.1)

        eurusd = obtener_velas(activo)
        # Estrategia EURUSD: 49 periodos, 3 desviaciones, 1 minuto
        sma_eurusd.append(eurusd[2])
        # Determinar tipo de vela
        if eurusd[2] >= eurusd[1]:
            tipo = "Verde"
        else:
            tipo = "Roja"
        # Calculo de las señales
        if tipo == "Verde":
            if alta_eurusd != 0 and eurusd[1] < baja_eurusd  and eurusd[2] > alta_eurusd and senal != 1:
                api.buy(inversion, activo, "turbo", "put")
                print "Operación bajista abierta a las:", datetime.now()
                senal = 1
        else:
            if baja_eurusd != 0 and eurusd[1] > alta_eurusd and eurusd[2] < baja_eurusd  and  senal != 2:
                api.buy(inversion, activo, "turbo", "call")
                print "Operación alcista abierta a las:", datetime.now()
                senal = 2
        # Calculo de las bandas bollinger
        if len(sma_eurusd) == periodos_eurusd:
            media = (sum(sma_eurusd)) / periodos_eurusd
            desviacion = np.std(np.array(sma_eurusd), ddof=1)
            alta_eurusd = media + (desviaciones_eurusd * desviacion)
            baja_eurusd = media - (desviaciones_eurusd * desviacion)
            del sma_eurusd[0]
         
        time.sleep(0.5)
    else:
        velas = conectar(activo) # Utilizar EURUSD para establecer la conexión siempre
        print "Conexión establecida\n"
        capital_inicial = api.profile.balance

