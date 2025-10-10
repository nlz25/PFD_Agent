import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Send, Bot, FileText, Terminal, X } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import SessionList from './SessionList'
import FileExplorer from './FileExplorer'
import { ShellTerminal } from './ShellTerminal'
import { ResizablePanel } from './ResizablePanel'
import { useAgentConfig } from '../hooks/useAgentConfig'
import { MessageAnimation, LoadingDots } from './MessageAnimation'
import { MemoizedMessage } from './MemoizedMessage'
import axios from 'axios'

const API_BASE_URL = ''  // Use proxy in vite config

interface MessageAttachment {
  name: string
  size?: number
  type?: string
  local_path?: string
}

interface Message {
  id: string
  role: 'user' | 'assistant' | 'tool'
  content: string
  timestamp: Date
  tool_name?: string
  tool_status?: string
  isStreaming?: boolean
  attachments?: MessageAttachment[]
}

interface Session {
  id: string
  title: string
  created_at: string
  last_message_at: string
  message_count: number
}

interface FileNode {
  name: string
  path: string
  type: 'file' | 'directory'
  children?: FileNode[]
  isExpanded?: boolean
  size?: number
  modified?: string
}

const ChatInterface: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([])
  const [sessions, setSessions] = useState<Session[]>([])
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null)
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [showLoadingDelay, setShowLoadingDelay] = useState(false)
  const [isCreatingSession, setIsCreatingSession] = useState(false)
  const [fileTree, setFileTree] = useState<FileNode[]>([])
  const [showFileExplorer, setShowFileExplorer] = useState(false)
  const [showShellTerminal, setShowShellTerminal] = useState(false)
  const [shellOutput, setShellOutput] = useState<Array<{ type: 'command' | 'output' | 'error'; content: string; timestamp: Date }>>([]) 
  const [selectedFiles, setSelectedFiles] = useState<File[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const messageIdef = useRef<Set<string>>(new Set())
  const loadingTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  
  // Load agent configuration
  const { config, loading: configLoading } = useAgentConfig()

  const scrollToBottom = useCallback(() => {
    setTimeout(() => {
      messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'nearest' })
      const scrollContainer = messagesEndRef.current?.parentElement?.parentElement
      if (scrollContainer) {
        const targetScroll = scrollContainer.scrollHeight - scrollContainer.clientHeight
        scrollContainer.scrollTo({
          top: targetScroll,
          behavior: 'smooth'
        })
      }
    }, 100)
  }, [])

  const loadFileTree = useCallback(async () => {
    try {
      const outputDir = config.files?.outputDirectory || 'output'
      const [outputResult, uploadedResult] = await Promise.allSettled([
        axios.get(`${API_BASE_URL}/api/files/tree`, { params: { path: outputDir } }),
        axios.get(`${API_BASE_URL}/api/files/tree`, { params: { path: 'uploaded_files' } })
      ])

      const tree: FileNode[] = []

      if (uploadedResult.status === 'fulfilled') {
        tree.push({
          name: 'uploaded_files',
          path: 'uploaded_files',
          type: 'directory',
          isExpanded: false,
          children: uploadedResult.value.data || []
        })
      }

      if (outputResult.status === 'fulfilled') {
        tree.push({
          name: outputDir,
          path: outputDir,
          type: 'directory',
          isExpanded: true,
          children: outputResult.value.data || []
        })
      } else {
        tree.push({
          name: outputDir,
          path: outputDir,
          type: 'directory',
          isExpanded: true,
          children: []
        })
      }

      setFileTree(tree)
    } catch (error) {
      console.error('Error loading file tree:', error)
      setFileTree([{
        name: 'output',
        path: 'output',
        type: 'directory',
        isExpanded: true,
        children: []
      }])
    }
  }, [config.files?.outputDirectory])

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading, scrollToBottom])

  // 延迟显示加载动画，避免闪烁
  useEffect(() => {
    if (isLoading) {
      loadingTimeoutRef.current = setTimeout(() => {
        setShowLoadingDelay(true)
      }, 200) // 200ms 延迟
    } else {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current)
      }
      setShowLoadingDelay(false)
    }
    
    return () => {
      if (loadingTimeoutRef.current) {
        clearTimeout(loadingTimeoutRef.current)
      }
    }
  }, [isLoading])

  const [ws, setWs] = useState<WebSocket | null>(null)
  const [connectionStatus, setConnectionStatus] = useState<'disconnected' | 'connecting' | 'connected'>('disconnected')

  useEffect(() => {
    // Load initial file tree
    loadFileTree()
    
    // Keep track of current websocket instance
    let currentWebSocket: WebSocket | null = null
    let reconnectTimeout: NodeJS.Timeout | null = null
    
    // Connect to WebSocket
    const connectWebSocket = () => {
      // Clean up any existing connection
      if (currentWebSocket?.readyState === WebSocket.OPEN || currentWebSocket?.readyState === WebSocket.CONNECTING) {
        currentWebSocket.close()
      }
      
      setConnectionStatus('connecting')
      // 动态获取 WebSocket URL，支持代理和远程访问
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
      const host = window.location.hostname
      const port = window.location.port
      
      // 如果是通过代理访问，使用当前页面的 host
      let wsUrl = `${protocol}//${host}`
      if (port) {
        wsUrl += `:${port}`
      }
      wsUrl += '/ws'
      
      console.log('Connecting to WebSocket:', wsUrl)
      const websocket = new WebSocket(wsUrl)
      currentWebSocket = websocket
      
      websocket.onopen = () => {
        console.log('WebSocket connected')
        setConnectionStatus('connected')
        setWs(websocket)
      }
      
      websocket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data)
          console.log('Received WebSocket message:', data)
          handleWebSocketMessage(data)
        } catch (error) {
          console.error('WebSocket message error:', error)
        }
      }
      
      websocket.onerror = (error) => {
        console.error('WebSocket error:', error)
        setConnectionStatus('disconnected')
      }
      
      websocket.onclose = () => {
        setConnectionStatus('disconnected')
        setWs(null)
        // Only reconnect if this is the current websocket
        if (websocket === currentWebSocket) {
          // Reconnect after 3 seconds
          reconnectTimeout = setTimeout(connectWebSocket, 3000)
        }
      }
    }
    
    connectWebSocket()
    
    return () => {
      // Clean up on unmount
      if (reconnectTimeout) {
        clearTimeout(reconnectTimeout)
      }
      if (currentWebSocket) {
        currentWebSocket.close()
      }
    }
  }, [loadFileTree])

  

  // Session management functions
  const handleCreateSession = useCallback(async () => {
    if (ws && connectionStatus === 'connected' && !isCreatingSession) {
      setIsCreatingSession(true)
      // 清空当前消息
      setMessages([])
      ws.send(JSON.stringify({ type: 'create_session' }))
      // 设置超时，避免永久等待
      setTimeout(() => {
        setIsCreatingSession(false)
      }, 3000)
    }
  }, [ws, connectionStatus, isCreatingSession])

  const handleSelectSession = useCallback(async (sessionId: string) => {
    if (ws && connectionStatus === 'connected') {
      ws.send(JSON.stringify({ 
        type: 'switch_session',
        session_id: sessionId 
      }))
    }
  }, [ws, connectionStatus])

  const handleDeleteSession = useCallback(async (sessionId: string) => {
    if (ws && connectionStatus === 'connected') {
      ws.send(JSON.stringify({ 
        type: 'delete_session',
        session_id: sessionId 
      }))
    }
  }, [ws, connectionStatus])

  const resetFileInput = () => {
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  const readFileAsBase64 = (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => {
        const result = reader.result
        if (typeof result === 'string') {
          const base64 = result.includes(',') ? result.split(',')[1] : result
          resolve(base64)
        } else if (result instanceof ArrayBuffer) {
          const bytes = new Uint8Array(result)
          let binary = ''
          bytes.forEach(byte => {
            binary += String.fromCharCode(byte)
          })
          resolve(btoa(binary))
        } else {
          reject(new Error('Unsupported file result'))
        }
      }
      reader.onerror = () => reject(reader.error || new Error('文件读取失败'))
      reader.readAsDataURL(file)
    })
  }

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    if (!files.length) {
      return
    }
    setSelectedFiles(prev => [...prev, ...files])
  }

  const handleRemoveFile = (index: number) => {
    setSelectedFiles(prev => prev.filter((_, i) => i !== index))
    resetFileInput()
  }

  const handleFileUploadClick = () => {
    fileInputRef.current?.click()
  }

  const handleSend = async () => {
    const trimmedInput = input.trim()
    if (!trimmedInput && selectedFiles.length === 0) return
    if (!ws || connectionStatus !== 'connected') {
      alert('未连接到服务器，请稍后重试')
      return
    }

    setIsLoading(true)

    const attachments: MessageAttachment[] = selectedFiles.map(file => ({
      name: file.name,
      size: file.size,
      type: file.type || 'application/octet-stream'
    }))

    let filePayloads: Array<{
      name: string
      type: string
      data: string
      size: number
    }> = []

    if (selectedFiles.length > 0) {
      try {
        filePayloads = await Promise.all(
          selectedFiles.map(async (file) => {
            const base64Data = await readFileAsBase64(file)
            return {
              name: file.name,
              type: file.type || 'application/octet-stream',
              data: base64Data,
              size: file.size
            }
          })
        )
      } catch (error) {
        console.error('读取文件失败', error)
        setIsLoading(false)
        alert('读取文件失败，请重试')
        return
      }
    }

    const displayContent = trimmedInput || (filePayloads.length
      ? filePayloads.map(file => `上传文件：${file.name}`).join('\n')
      : '')

    const newMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: displayContent,
      timestamp: new Date(),
      attachments
    }

    setMessages(prev => [...prev, newMessage])
    setInput('')
    setSelectedFiles([])
    resetFileInput()

    scrollToBottom()

    const payload: {
      type: string
      content: string
      files?: Array<{
        name: string
        type: string
        data: string
        size: number
      }>
    } = {
      type: 'message',
      content: trimmedInput
    }

    if (filePayloads.length > 0) {
      payload.files = filePayloads
      if (filePayloads.length === 1) {
        (payload as any).file = filePayloads[0]
      }
    }

    ws.send(JSON.stringify(payload))
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      void handleSend()
    }
  }

  const handleWebSocketMessage = useCallback((data: any) => {
    const { type, content, timestamp, id } = data
    
    // 如果消息有ID，检查是否已经处理过
    if (id && messageIdef.current.has(id)) {
      return
    }
    if (id) {
      messageIdef.current.add(id)
    }
    
    // Handle shell command responses
    if (type === 'shell_output') {
      setShellOutput(prev => [...prev, {
        type: 'output',
        content: data.output || '',
        timestamp: new Date()
      }])
      return
    }
    
    if (type === 'shell_error') {
      setShellOutput(prev => [...prev, {
        type: 'error',
        content: data.error || 'Command execution error',
        timestamp: new Date()
      }])
      return
    }
    
    if (type === 'sessions_list') {
      // 更新会话列表
      setSessions(data.sessions || [])
      setCurrentSessionId(data.current_session_id)
      setIsCreatingSession(false)
      return
    }

    if (type === 'files_updated') {
      loadFileTree()
      return
    }
    
    if (type === 'session_messages') {
      // 加载会话历史消息
      const messages = data.messages || []
      const normalizedMessages: Message[] = messages.map((msg: any) => ({
        id: msg.id,
        role: msg.role,
        content: msg.content || '',
        timestamp: msg.timestamp ? new Date(msg.timestamp) : new Date(),
        tool_name: msg.tool_name,
        tool_status: msg.tool_status,
        attachments: Array.isArray(msg.attachments)
          ? msg.attachments.map((item: any) => ({
              name: item?.name ?? '文件',
              size: typeof item?.size === 'number' ? item.size : undefined,
              type: item?.type
            }))
          : []
      }))
      setMessages(normalizedMessages)
      // 清除消息ID缓存，避免重复
      messageIdef.current.clear()
      normalizedMessages.forEach((msg: Message) => {
        if (msg.id) {
          messageIdef.current.add(msg.id)
        }
      })
      setIsCreatingSession(false)
      return
    }
    
    if (type === 'user') {
      // Skip echoed user messages
      return
    }
    
    if (type === 'tool') {
      // Tool execution status
      const { tool_name, status, is_long_running, result } = data
      let content = ''
      
      if (status === 'executing') {
        const icon = is_long_running ? '⏳' : '🔧'
        content = `${icon} 正在执行工具: **${tool_name}**${is_long_running ? ' (长时间运行)' : ''}`
      } else if (status === 'completed') {
        if (result) {
          // 保留原始格式，包括换行符
          content = `✅ 工具执行完成: **${tool_name}**\n\`\`\`json\n${result}\n\`\`\``
        } else {
          content = `✅ 工具执行完成: **${tool_name}**`
        }
      } else {
        content = `📊 工具状态更新: **${tool_name}** - ${status}`
      }
      
      const toolMessage: Message = {
        id: id || `tool-${Date.now()}`,
        role: 'tool' as const,
        content,
        timestamp: new Date(timestamp || Date.now()),
        tool_name,
        tool_status: status,
        attachments: []
      }
      
      // 使用函数式更新来避免消息重复
      setMessages(prev => {
        // 检查是否已经存在相同ID的消息
        if (prev.some(m => m.id === toolMessage.id)) {
          return prev
        }
        return [...prev, toolMessage]
      })
      // 工具消息后滚动到底部
      scrollToBottom()
      if (status === 'completed') {
        loadFileTree()
      }
      return
    }
    
    if (type === 'assistant' || type === 'response') {
      const rawAttachments = Array.isArray(data.attachments) ? data.attachments : []
      const attachmentList: MessageAttachment[] = rawAttachments.map((item: any) => ({
        name: item?.name ?? '文件',
        size: typeof item?.size === 'number' ? item.size : undefined,
        type: item?.type,
        local_path: item?.local_path
      }))

      const assistantMessage: Message = {
        id: id || `assistant-${Date.now()}`,
        role: 'assistant',
        content: content || '',
        timestamp: new Date(timestamp || Date.now()),
        attachments: attachmentList
      }

      // 使用函数式更新来避免消息重复
      setMessages(prev => {
        // 检查是否已经存在相同ID的消息
        if (prev.some(m => m.id === assistantMessage.id)) {
          return prev
        }
        return [...prev, assistantMessage]
      })
      // 收到新消息后滚动到底部
      scrollToBottom()
    }

    if (type === 'complete') {
      setIsLoading(false)
      // 加载完成后滚动到底部
      scrollToBottom()
    }
    
    if (type === 'error') {
      const errorMessage: Message = {
        id: `error-${Date.now()}`,
        role: 'assistant',
        content: `❌ 错误: ${content}`,
        timestamp: new Date(),
        attachments: []
      }
      setMessages(prev => [...prev, errorMessage])
      setIsLoading(false)
    }
  }, [loadFileTree, scrollToBottom])

  return (
    <div className="flex h-screen bg-gray-50 dark:bg-gray-900">
      {/* Session List Sidebar */}
      <ResizablePanel
        direction="horizontal"
        minSize={200}
        maxSize={400}
        defaultSize={280}
        className="border-r border-gray-200 dark:border-gray-700"
      >
        <SessionList
          sessions={sessions}
          currentSessionId={currentSessionId}
          onCreateSession={handleCreateSession}
          onSelectSession={handleSelectSession}
          onDeleteSession={handleDeleteSession}
        />
      </ResizablePanel>

      {/* Main Content Area */}
      <div className="flex-1 flex min-w-0">
        {/* Chat Area */}
        <div className="flex-1 flex flex-col min-w-0 bg-gradient-to-br from-gray-50 via-white to-gray-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 aurora-bg">
        {/* Header */}
        <div className="px-4 py-3 border-b border-gray-200/50 dark:border-gray-700/50 glass-premium glass-glossy flex items-center justify-between">
          <div className="flex items-center gap-3">
            <h1 className="text-lg font-semibold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              {config.ui?.title || 'Agent'}
            </h1>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setShowShellTerminal(!showShellTerminal)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors btn-animated"
            >
              <Terminal className="w-4 h-4" />
              {showShellTerminal ? '隐藏终端' : '显示终端'}
            </button>
            <button
              onClick={() => setShowFileExplorer(!showFileExplorer)}
              className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors btn-animated"
            >
              <FileText className="w-4 h-4" />
              {showFileExplorer ? '隐藏文件' : '查看文件'}
            </button>
            <div className={`flex items-center gap-2 px-3 py-1 rounded-full text-xs font-medium ${
              connectionStatus === 'connected' 
                ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400' 
                : connectionStatus === 'connecting'
                ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-400'
                : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-400'
            }`}>
              <div className={`w-2 h-2 rounded-full ${
                connectionStatus === 'connected' ? 'bg-green-500' : 
                connectionStatus === 'connecting' ? 'bg-yellow-500 animate-pulse' : 
                'bg-red-500'
              }`} />
              <span>
                {connectionStatus === 'connected' ? '已连接' : 
                 connectionStatus === 'connecting' ? '连接中...' : 
                 '未连接'}
              </span>
            </div>
          </div>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6 relative">
          <div className="max-w-4xl mx-auto space-y-6 h-full">
            {messages.length === 0 ? (
              <div className="absolute inset-0 flex items-center justify-center">
                <div className="text-center">
                  <Bot className="w-16 h-16 mx-auto mb-4 text-gray-300 dark:text-gray-600" />
                  <h3 className="text-lg font-medium text-gray-600 dark:text-gray-400 mb-2">
                    欢迎使用 {config.agent?.name || 'Agent'}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-500">
                    {config.agent?.welcomeMessage || '输入您的数据文件路径，开始符号回归分析'}
                  </p>
                </div>
              </div>
            ) : (
              <AnimatePresence initial={false} mode="popLayout">
                {messages.map((message, index) => (
                  <motion.div
                    key={message.id}
                    layout="position"
                    initial={index === messages.length - 1 ? { opacity: 0, y: 20 } : false}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -20 }}
                    transition={{ duration: 0.2 }}
                    className={`flex gap-4 ${
                      message.role === 'user' ? 'justify-end' : 'justify-start'
                    }`}
                  >
                    <MemoizedMessage
                      id={message.id}
                      role={message.role}
                      content={message.content}
                      timestamp={message.timestamp}
                      isLastMessage={index === messages.length - 1}
                      isStreaming={message.isStreaming}
                      attachments={message.attachments ?? []}
                    />
                  </motion.div>
                ))}
              </AnimatePresence>
            )}
            
            {showLoadingDelay && (
              <MessageAnimation isNew={true} type="assistant">
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="flex gap-4"
                >
                  <div className="flex-shrink-0">
                    <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center shadow-lg">
                      <Bot className="w-5 h-5 text-white" />
                    </div>
                  </div>
                  <div className="bg-white dark:bg-gray-800 rounded-2xl px-4 py-3 shadow-sm border border-gray-200 dark:border-gray-700">
                    <LoadingDots />
                  </div>
                </motion.div>
              </MessageAnimation>
            )}
            
            {/* 底部垫高，确保最后一条消息不贴底 */}
            <div className="h-24" />
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 dark:border-gray-700 glass-premium p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex flex-col gap-3">
              <div className="flex gap-3">
                <div className="flex flex-1 items-end gap-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    className="hidden"
                    onChange={handleFileChange}
                  />
                  <button
                    type="button"
                    onClick={handleFileUploadClick}
                    className="flex items-center gap-2 px-3 py-2 rounded-xl border border-dashed border-blue-400 text-sm font-medium text-blue-600 dark:text-blue-300 hover:bg-blue-50 dark:hover:bg-blue-900/30 transition-colors"
                  >
                    <FileText className="w-4 h-4" />
                    上传文件
                  </button>
                  <textarea
                    ref={inputRef}
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="输入消息..."
                    className="flex-1 resize-none rounded-xl border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 px-4 py-3 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 dark:focus:ring-blue-400 transition-all input-animated glow"
                    rows={1}
                    style={{
                      minHeight: '48px',
                      maxHeight: '200px'
                    }}
                    onInput={(e) => {
                      const target = e.target as HTMLTextAreaElement
                      target.style.height = 'auto'
                      target.style.height = `${target.scrollHeight}px`
                    }}
                  />
                </div>
                <button
                  type="button"
                  onClick={handleSend}
                  disabled={(!input.trim() && selectedFiles.length === 0) || isLoading || connectionStatus !== 'connected'}
                  className="px-4 py-2 bg-gradient-to-r from-blue-500 to-blue-600 text-white rounded-xl font-medium hover:from-blue-600 hover:to-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-lg hover:shadow-xl flex items-center gap-2 btn-animated liquid-button"
                >
                  <Send className="w-4 h-4" />
                  发送
                </button>
              </div>
              {selectedFiles.length > 0 && (
                <div className="flex flex-wrap gap-2">
                  {selectedFiles.map((file, index) => (
                    <div
                      key={`${file.name}-${file.lastModified}-${index}`}
                      className="flex items-center gap-2 rounded-xl border border-blue-200 dark:border-blue-700 bg-blue-50/80 dark:bg-blue-900/40 px-3 py-2 text-sm text-blue-700 dark:text-blue-200"
                    >
                      <FileText className="w-4 h-4" />
                      <span className="flex-1 truncate max-w-[200px]">{file.name}</span>
                      <button
                        type="button"
                        onClick={() => handleRemoveFile(index)}
                        className="p-1 rounded-full hover:bg-blue-200/70 dark:hover:bg-blue-800/70 transition-colors"
                        aria-label="移除文件"
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
        </div>
        
        {/* File Explorer Sidebar */}
        {showFileExplorer && (
          <ResizablePanel
            direction="horizontal"
            minSize={400}
            maxSize={800}
            defaultSize={600}
            className="border-l border-gray-200 dark:border-gray-700"
            resizeBarPosition="start"
          >
            <FileExplorer
              isOpen={showFileExplorer}
              onClose={() => setShowFileExplorer(false)}
              fileTree={fileTree}
              onFileTreeUpdate={setFileTree}
              onLoadFileTree={loadFileTree}
            />
          </ResizablePanel>
        )}
      </div>
      
      {/* Shell Terminal */}
      <ShellTerminal
        isOpen={showShellTerminal}
        onClose={() => setShowShellTerminal(false)}
        onExecuteCommand={(command) => {
          if (command === '__clear__') {
            setShellOutput([])
            return
          }
          
          // Add command to output
          setShellOutput(prev => [...prev, {
            type: 'command',
            content: command,
            timestamp: new Date()
          }])
          
          // Send command to server
          if (ws && connectionStatus === 'connected') {
            ws.send(JSON.stringify({
              type: 'shell_command',
              command: command
            }))
          } else {
            setShellOutput(prev => [...prev, {
              type: 'error',
              content: 'Not connected to server',
              timestamp: new Date()
            }])
          }
        }}
        output={shellOutput}
      />
    </div>
  )
}

export default ChatInterface
