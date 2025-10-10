import React from 'react';
import { Bot, User, FileText } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { createCodeComponent } from './EnhancedCodeBlock';
import { MemoizedMarkdown } from './MemoizedMarkdown';
import { StreamingText } from './MessageAnimation';

interface MessageAttachment {
  name: string;
  size?: number;
  type?: string;
  local_path?: string;
}

interface MessageProps {
  id: string;
  role: 'user' | 'assistant' | 'tool';
  content: string;
  timestamp: Date;
  isLastMessage?: boolean;
  isStreaming?: boolean;
  attachments?: MessageAttachment[];
}

const formatFileSize = (size?: number): string => {
  if (typeof size !== 'number' || Number.isNaN(size)) {
    return '';
  }
  if (size >= 1024 * 1024) {
    return `${(size / (1024 * 1024)).toFixed(1)} MB`;
  }
  if (size >= 1024) {
    return `${(size / 1024).toFixed(1)} KB`;
  }
  return `${size} B`;
};

const attachmentsEqual = (a: MessageAttachment[] = [], b: MessageAttachment[] = []): boolean => {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i += 1) {
    const attA = a[i];
    const attB = b[i];
    if (attA.name !== attB.name || attA.size !== attB.size || attA.type !== attB.type) {
      return false;
    }
  }
  return true;
};

export const MemoizedMessage = React.memo<MessageProps>(({
  id,
  role,
  content,
  timestamp,
  isLastMessage = false,
  isStreaming = false,
  attachments = []
}) => {
  const attachmentList = attachments ?? [];
  const hasText = Boolean(content && content.trim().length > 0);
  const containerClasses = `max-w-[80%] ${role === 'user' ? 'order-1' : ''} flex flex-col ${role === 'user' ? 'items-end' : 'items-start'}`;
  const attachmentsJustify = role === 'user' ? 'justify-end' : 'justify-start';
  const timestampAlignment = role === 'user' ? 'self-end' : 'self-start';

  return (
    <>
      {role !== 'user' && (
        <div className="flex-shrink-0">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-500 flex items-center justify-center shadow-lg">
            <Bot className="w-5 h-5 text-white" />
          </div>
        </div>
      )}

      <div className={containerClasses}>
        {hasText && (
          <div className={`rounded-2xl px-4 py-3 shadow-sm ${
            role === 'user'
              ? 'bg-gradient-to-r from-blue-500 to-blue-600 text-white'
              : role === 'tool'
              ? 'bg-gray-100 dark:bg-gray-800 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-700 glass-premium'
              : 'bg-white dark:bg-gray-800 text-gray-800 dark:text-gray-200 border border-gray-200 dark:border-gray-700 glass-premium shadow-depth'
          }`}>
            {role === 'tool' ? (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <MemoizedMarkdown>
                  {content}
                </MemoizedMarkdown>
              </div>
            ) : role === 'assistant' ? (
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  components={{
                    code: createCodeComponent(),
                    a({ node, children, href, ...props }: any) {
                      return (
                        <a
                          href={href}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-blue-600 hover:text-blue-800 dark:text-blue-400 dark:hover:text-blue-300 underline"
                          {...props}
                        >
                          {children}
                        </a>
                      );
                    },
                    p({ children }: any) {
                      if (isLastMessage && isStreaming) {
                        return (
                          <p>
                            <StreamingText
                              text={String(children)}
                              isStreaming={true}
                            />
                          </p>
                        );
                      }
                      return <p>{children}</p>;
                    }
                  }}
                >
                  {content}
                </ReactMarkdown>
              </div>
            ) : (
              <p className="text-sm whitespace-pre-wrap">{content}</p>
            )}
          </div>
        )}

        {attachmentList.length > 0 && (
          <div className={`flex flex-wrap gap-4 ${attachmentsJustify} ${hasText ? 'mt-3' : ''}`}>
            {attachmentList.map((file, index) => {
              const sizeLabel = formatFileSize(file.size);
              return (
                <div key={`${file.name}-${index}`} className="flex flex-col items-center gap-1">
                  <div className="w-12 h-12 rounded-xl border border-blue-200 dark:border-blue-500 bg-blue-50 dark:bg-blue-900/30 flex items-center justify-center text-blue-600 dark:text-blue-300 shadow-sm">
                    <FileText className="w-6 h-6" />
                  </div>
                  <span className="max-w-[140px] text-xs text-blue-600 dark:text-blue-200 break-words text-center">
                    {file.name}
                  </span>
                  {sizeLabel && (
                    <span className="text-[10px] text-gray-500 dark:text-gray-400">
                      {sizeLabel}
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        )}

        <p className={`text-xs text-gray-500 dark:text-gray-400 mt-2 px-1 ${timestampAlignment}`}>
          {timestamp.toLocaleTimeString('zh-CN')}
        </p>
      </div>

      {role === 'user' && (
        <div className="flex-shrink-0 order-2">
          <div className="w-8 h-8 rounded-full bg-gradient-to-br from-gray-600 to-gray-700 flex items-center justify-center shadow-lg">
            <User className="w-5 h-5 text-white" />
          </div>
        </div>
      )}
    </>
  );
}, (prevProps, nextProps) => (
  prevProps.id === nextProps.id &&
  prevProps.content === nextProps.content &&
  prevProps.isStreaming === nextProps.isStreaming &&
  prevProps.isLastMessage === nextProps.isLastMessage &&
  attachmentsEqual(prevProps.attachments ?? [], nextProps.attachments ?? [])
));

MemoizedMessage.displayName = 'MemoizedMessage';
