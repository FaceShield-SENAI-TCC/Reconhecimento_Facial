import os
import subprocess


def kill_processes_on_ports(ports):
    """Mata processos usando as portas especificadas"""
    for port in ports:
        try:
            # Encontrar PID usando a porta
            result = subprocess.run(
                f"netstat -ano | findstr :{port} | findstr LISTENING",
                capture_output=True,
                text=True,
                shell=True
            )

            lines = result.stdout.strip().split('\n')
            for line in lines:
                if line.strip():
                    parts = line.split()
                    pid = parts[-1]

                    # Matar o processo
                    os.system(f"taskkill /PID {pid} /F")
                    print(f"✅ Processo {pid} usando porta {port} foi terminado")

        except Exception as e:
            print(f"❌ Erro ao verificar porta {port}: {e}")


if __name__ == '__main__':
    ports = [5005, 3000, 3001, 3002, 3003, 8080, 8000]
    print("🔍 Procurando processos usando as portas...")
    kill_processes_on_ports(ports)
    print("✅ Concluído!")