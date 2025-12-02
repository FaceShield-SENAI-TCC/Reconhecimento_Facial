import asyncio
from flask import Blueprint, request, jsonify
from common.locker_controller import LockerController

iot_bp = Blueprint('iot_bp', __name__)

@iot_bp.route('/iot/control', methods=['POST', 'OPTIONS'])
def iot_control():
    """Endpoint para controle IOT da trava - só fecha quando armário fechar"""
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        return response

    try:
        data = request.get_json()
        if not data:
            return jsonify({'success': False, 'error': 'Dados JSON necessários'}), 400

        command = data.get('command', '').upper()
        print(f"COMANDO IOT RECEBIDO: {command}")

        if command == "ABRIR_TRAVA_IOT":
            print("INICIANDO ABERTURA DE TRAVA NO MODO IOT...")

            # CORREÇÃO 1: Criar INSTÂNCIA de LockerController
            # CORREÇÃO 2: Usar método existente abrir_trava_qrcode()
            locker_controller = LockerController()

            # Executa assincronamente
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            sucesso = loop.run_until_complete(locker_controller.abrir_trava_qrcode())
            loop.close()

            if sucesso:
                print("TRAVA ABERTA NO MODO IOT - AGUARDANDO FECHAMENTO DO ARMARIO")
                return jsonify({
                    'success': True,
                    'message': 'Trava liberada no modo IOT'
                })
            else:
                print("FALHA AO ABRIR TRAVA NO MODO IOT")
                return jsonify({
                    'success': False,
                    'error': 'Falha ao comunicar com a ESP32'
                }), 500

        else:
            return jsonify({
                'success': False,
                'error': f'Comando não reconhecido: {command}'
            }), 400

    except Exception as e:
        print(f"ERRO NO CONTROLE IOT: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Erro interno: {str(e)}'
        }), 500