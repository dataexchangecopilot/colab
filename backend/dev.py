from sympy import false, true
import uvicorn

PORT="${PORT:-8080}"
uvicorn.run("open_webui.main:app",
            port = 8080,
            host = "0.0.0.0",
            forwarded_allow_ips="*",
            reload =false
)