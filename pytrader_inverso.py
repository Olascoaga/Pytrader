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
    velas = None
    while velas is None:
        print "Intento de conexión..."
        api = IQOptionAPI("iqoption.com", "usuario", "contraseña")
        api.connect()
        api.getcandles(id_activo, 1)
        velas = api.candles.candles_data
    return velas


def obtener_velas(id_activo):

    if id_activo == 1:
        activo = "EURUSD"
    else:
        activo = "EURGBP"

    try:
        # print "Peticion de velas:", "-", datetime.now()
        api.getcandles(id_activo, 1)  # produce una excepción cuándo el websocket pierde conexión con el servidor
    except:
        print "¡Desconexión!", datetime.now()

    time.sleep(0.08)
    if api.candles.current_candle.candle_close != 0:
        cierre = api.candles.current_candle.candle_close #/ 1000000.0
    else:
        cierre = api.candles.candles_data[-2][2] #/ 1000000.0
    vela = [api.candles.first_candle.candle_time,
            api.candles.first_candle.candle_open, #/ 1000000.0,
            cierre]
    """
    print "Hora apertura:", activo, datetime.fromtimestamp(api.candles.first_candle.candle_time)
    print "Hora cierre:", datetime.fromtimestamp(api.candles.current_candle.candle_time)
    print "Valor cierre:", cierre, "\n"
    """

    return vela  # Para vela: [0] = fecha, [1] = Apertura, [2] = Cierre

# Constantes
hora_cero = True
chances = 15
velas = None
activo = 1
api.timesync.expiration_time = 1
# Configuración estrategia EURUSD
sma_eurusd = []
periodos_eurusd = 10
desviaciones_eurusd = 1
alta_eurusd = 0
baja_eurusd = 0
# Configuración estrategia EURGBP
sma_eurgbp = []
senal = 0  # 0 = sin senal, 1 = alza, 2 = baja
while True:
    if velas is not None:
        if hora_cero:
            print "Capital Inicial:", capital_inicial
            while datetime.now().second != 0:
                time.sleep(0.01)
                hora_cero = False

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
            if alta_eurusd != 0 and eurusd[2] > alta_eurusd and senal != 1:
                balance = api.profile.balance
                inversion = round(balance / chances, 2)
                if inversion < 1:
                    inversion = 1
                api.buy(inversion, activo, "turbo", "put")
                print "Operación bajista abierta a las:", datetime.now()
                senal = 1
        else:
            if baja_eurusd != 0 and eurusd[2] < baja_eurusd and senal != 2:
                balance = api.profile.balance
                inversion = round(balance / chances, 2)
                if inversion < 1:
                    inversion = 1
                api.buy(inversion, activo, "turbo", "call")
                print "Operación alcista abierta a las:", datetime.now()
                senal = 2
        # Calculo de las bandas bollinger
        if len(sma_eurusd) == periodos_eurusd:
            media = (sum(sma_eurusd)) / periodos_eurusd
            desviacion = np.std(np.array(sma_eurusd))
            alta_eurusd = media + (desviaciones_eurusd * desviacion)
            baja_eurusd = media - (desviaciones_eurusd * desviacion)
            del sma_eurusd[0]

        #api.set_session_cookies()
        capital_actual = api.profile.balance
        if capital_actual >= (capital_inicial * 1.30):
            print "¡Meta alcanzada!"
            break
        time.sleep(10)
    else:
        velas = conectar(activo) # Utilizar EURUSD para establecer la conexión siempre
        print "Conexión establecida\n"

    capital_inicial = api.profile.balance
