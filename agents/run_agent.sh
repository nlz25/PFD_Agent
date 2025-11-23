export OPENAI_API_KEY=sk-78457c528a7b4751ac461bd597467a80
export OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1

python3 /mnt/t/programming/pfd_agent/pfd-agent-dev/tools/database/server.py --port=50002 &
python3 /mnt/t/programming/pfd_agent/pfd-agent-dev/tools/abacus/server.py --port=50001 &

adk web


