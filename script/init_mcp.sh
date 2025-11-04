#!/usr/bin/env bash
pfd-agent --port 50001 > pfd-agent.log 2>&1 &   # start in background
database-agent --port 50002 > database-agent.log 2>&1 &

# optionally detach from shell so they survive logout
#disown