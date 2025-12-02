import logging
from flask import Blueprint, request, jsonify
from common.event_loop_manager import EventLoopManager

logger = logging.getLogger(__name__)
iot_bp = Blueprint('iot_bp', __name__)


@iot_bp.route('/iot/control', methods=['POST', 'OPTIONS'])
def iot_control():
    """
    Endpoint para controle IoT da ESP32
    Comandos aceitos:
    - ABRIR_TRAVA_IOT: Abre trava (modo QR Code)
    - ABRIR_TRAVA: Abre trava (modo Facial - 10s)
    - FECHAR_TRAVA: Fecha trava manualmente
    - STATUS: Retorna estado da trava
    - TESTAR_CONEXAO: Testa conexão
    """

    # Handle preflight CORS
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST, OPTIONS')
        return response

    try:
        # Verificar se é JSON
        if not request.is_json:
            logger.error(" Content-Type não é application/json")
            return jsonify({
                'success': False,
                'error': 'Content-Type deve ser application/json'
            }), 400

        # Obter dados JSON
        data = request.get_json()
        if not data:
            logger.error(" Dados JSON vazios ou inválidos")
            return jsonify({
                'success': False,
                'error': 'Dados JSON necessários'
            }), 400

        # Extrair comando
        command = data.get('command', '').upper().strip()
        logger.info(f" Comando IoT recebido: '{command}'")

        if not command:
            return jsonify({
                'success': False,
                'error': 'Campo "command" é obrigatório'
            }), 400

        # Importar aqui para evitar circular imports
        from common.locker_controller import LockerController

        # Criar instância do controlador
        locker_controller = LockerController()

        # Processar comando
        if command == "ABRIR_TRAVA_IOT":
            logger.info("===  ABRINDO TRAVA NO MODO IOT (QR CODE) ===")

            try:
                sucesso = EventLoopManager.run_async(
                    locker_controller.abrir_trava_qrcode()
                )

                if sucesso:
                    logger.info(" TRAVA ABERTA NO MODO IOT")
                    return jsonify({
                        'success': True,
                        'message': 'Trava liberada no modo IOT',
                        'status': 'aberta_iot'
                    })
                else:
                    logger.error(" FALHA AO ABRIR TRAVA NO MODO IOT")
                    return jsonify({
                        'success': False,
                        'error': 'Falha ao comunicar com a ESP32'
                    }), 500

            except Exception as e:
                logger.error(f" ERRO no controle IoT: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': f'Erro interno: {str(e)}'
                }), 500

        elif command == "ABRIR_TRAVA":
            logger.info("===  ABRINDO TRAVA NO MODO FACIAL ===")

            try:
                sucesso = EventLoopManager.run_async(
                    locker_controller.abrir_trava_aluno(999, "ALUNO")
                )

                if sucesso:
                    logger.info(" TRAVA ABERTA NO MODO FACIAL (10s)")
                    return jsonify({
                        'success': True,
                        'message': 'Trava liberada no modo Facial',
                        'status': 'aberta_facial'
                    })
                else:
                    logger.error(" FALHA AO ABRIR TRAVA NO MODO FACIAL")
                    return jsonify({
                        'success': False,
                        'error': 'Falha ao comunicar com a ESP32'
                    }), 500

            except Exception as e:
                logger.error(f" ERRO no controle IoT: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': f'Erro interno: {str(e)}'
                }), 500

        elif command == "FECHAR_TRAVA":
            logger.info("=== ⏹  FECHANDO TRAVA MANUALMENTE ===")

            try:
                sucesso = EventLoopManager.run_async(
                    locker_controller.fechar_trava()
                )

                if sucesso:
                    logger.info(" TRAVA FECHADA MANUALMENTE")
                    return jsonify({
                        'success': True,
                        'message': 'Trava fechada',
                        'status': 'fechada'
                    })
                else:
                    logger.error(" FALHA AO FECHAR TRAVA")
                    return jsonify({
                        'success': False,
                        'error': 'Falha ao comunicar com a ESP32'
                    }), 500

            except Exception as e:
                logger.error(f" ERRO no controle IoT: {str(e)}", exc_info=True)
                return jsonify({
                    'success': False,
                    'error': f'Erro interno: {str(e)}'
                }), 500

        elif command == "STATUS":
            logger.info("===  SOLICITANDO STATUS DA TRAVA ===")

            try:
                status = EventLoopManager.run_async(
                    locker_controller.verificar_estado_trava()
                )

                logger.info(f" Status: {status}")
                return jsonify({
                    'success': True,
                    'status': status,
                    'message': 'Status obtido com sucesso'
                })

            except Exception as e:
                logger.error(f" ERRO ao obter status: {str(e)}")
                return jsonify({
                    'success': False,
                    'error': f'Erro ao obter status: {str(e)}'
                }), 500

        elif command == "TESTAR_CONEXAO":
            logger.info("===  TESTANDO CONEXÃO COM ESP32 ===")

            try:
                # Tenta obter status como teste de conexão
                status = EventLoopManager.run_async(
                    locker_controller.verificar_estado_trava()
                )

                logger.info(f" Conexão OK! Status: {status}")
                return jsonify({
                    'success': True,
                    'connected': True,
                    'message': 'Conexão com ESP32 estabelecida',
                    'esp32_status': status
                })

            except Exception as e:
                logger.error(f" Conexão falhou: {str(e)}")
                return jsonify({
                    'success': False,
                    'connected': False,
                    'error': f'Falha na conexão: {str(e)}'
                }), 500

        else:
            logger.warning(f" Comando não reconhecido: {command}")
            return jsonify({
                'success': False,
                'error': f'Comando não reconhecido. Use: ABRIR_TRAVA_IOT, ABRIR_TRAVA, FECHAR_TRAVA, STATUS, TESTAR_CONEXAO'
            }), 400

    except Exception as e:
        logger.error(f" ERRO GERAL no endpoint IoT: {str(e)}", exc_info=True)
        return jsonify({
            'success': False,
            'error': f'Erro interno do servidor: {str(e)}'
        }), 500