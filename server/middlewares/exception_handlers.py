from fastapi import Request
from fastapi.responses import JSONResponse
from server.logger import logger

async def catch_exception_middleware(request:Request,call_next):
    try:
        return await call_next(request)
    except Exception as ext:
        logger.exception("Unhandled Exception")
        return JSONResponse(status_code=500,content={"error":str(ext)})