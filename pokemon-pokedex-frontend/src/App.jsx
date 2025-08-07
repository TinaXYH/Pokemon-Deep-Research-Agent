import React, { useState, useRef, useEffect } from 'react';
import { Send, Loader2, CheckCircle, Clock, AlertCircle } from 'lucide-react';
import './App.css';

const API_BASE_URL = 'http://localhost:5175';

function App() {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [currentPhase, setCurrentPhase] = useState(null);
  const [conversationId] = useState('default');
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const phases = [
    { id: 'query_analysis', name: 'Query Analysis', icon: '🔍' },
    { id: 'specification_confirmation', name: 'Specification Confirmation', icon: '⚙️' },
    { id: 'data_collection', name: 'Data Collection', icon: '📊' },
    { id: 'deep_analysis', name: 'Deep Analysis', icon: '🧠' },
    { id: 'report_generation', name: 'Report Generation', icon: '📝' }
  ];

  const getPhaseStatus = (phaseId) => {
    if (!currentPhase) return 'pending';
    if (currentPhase.phase === phaseId) {
      return currentPhase.status === 'completed' ? 'completed' : 'active';
    }
    const currentIndex = phases.findIndex(p => p.id === currentPhase.phase);
    const phaseIndex = phases.findIndex(p => p.id === phaseId);
    return phaseIndex < currentIndex ? 'completed' : 'pending';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userMessage = inputValue.trim();
    setInputValue('');
    setIsLoading(true);
    setCurrentPhase(null);

    // Add user message
    setMessages(prev => [...prev, {
      type: 'user',
      content: userMessage,
      timestamp: new Date().toISOString()
    }]);

    try {
      const response = await fetch(`${API_BASE_URL}/api/research`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: userMessage,
          conversation_id: conversationId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let finalReport = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value);
        const lines = chunk.split('\n');

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.phase === 'error') {
                throw new Error(data.message);
              }

              setCurrentPhase(data);

              if (data.phase === 'report_generation' && data.status === 'completed') {
                finalReport = data.data.report;
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }

      // Add final report
      if (finalReport) {
        setMessages(prev => [...prev, {
          type: 'assistant',
          content: finalReport,
          timestamp: new Date().toISOString()
        }]);
      }

    } catch (error) {
      console.error('Error:', error);
      setMessages(prev => [...prev, {
        type: 'error',
        content: `Error: ${error.message}. Make sure the backend is running on http://localhost:5175`,
        timestamp: new Date().toISOString()
      }]);
    } finally {
      setIsLoading(false);
      setCurrentPhase(null);
    }
  };

  const handleQuickQuery = (query) => {
    setInputValue(query);
  };

  const formatMessage = (content) => {
    // Simple markdown-like formatting
    return content
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
      <div className="container mx-auto px-4 py-8 max-w-4xl">
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Pokemon Deep Research Agent
          </h1>
          <p className="text-gray-600">
            Interactive Pokédex with AI-powered competitive analysis
          </p>
        </div>

        {/* Research Progress */}
        {isLoading && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4 flex items-center">
              <Loader2 className="animate-spin mr-2" size={20} />
              Deep Research in Progress
            </h3>
            <div className="space-y-3">
              {phases.map((phase) => {
                const status = getPhaseStatus(phase.id);
                return (
                  <div key={phase.id} className="flex items-center space-x-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                      status === 'completed' ? 'bg-green-100 text-green-600' :
                      status === 'active' ? 'bg-blue-100 text-blue-600' :
                      'bg-gray-100 text-gray-400'
                    }`}>
                      {status === 'completed' ? <CheckCircle size={16} /> :
                       status === 'active' ? <Loader2 className="animate-spin" size={16} /> :
                       <Clock size={16} />}
                    </div>
                    <span className={`flex-1 ${
                      status === 'completed' ? 'text-green-600' :
                      status === 'active' ? 'text-blue-600' :
                      'text-gray-400'
                    }`}>
                      {phase.icon} {phase.name}
                    </span>
                    {currentPhase?.phase === phase.id && currentPhase?.message && (
                      <span className="text-sm text-gray-500">
                        {currentPhase.message}
                      </span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Quick Actions */}
        {messages.length === 0 && !isLoading && (
          <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
            <h3 className="text-lg font-semibold mb-4">Quick Research Topics</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {[
                "Tell me about Garchomp's competitive viability",
                "Build me a team around Pikachu",
                "Compare Charizard and Blastoise",
                "What movesets does Mewtwo use?",
                "Analyze the current OU meta",
                "Recommend Pokemon for beginners"
              ].map((query, index) => (
                <button
                  key={index}
                  onClick={() => handleQuickQuery(query)}
                  className="text-left p-3 bg-gray-50 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <span className="text-sm text-gray-700">{query}</span>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Messages */}
        <div className="bg-white rounded-lg shadow-lg mb-6 max-h-96 overflow-y-auto">
          <div className="p-6 space-y-4">
            {messages.length === 0 && !isLoading && (
              <div className="text-center text-gray-500 py-8">
                <div className="text-6xl mb-4">🎮</div>
                <p>Ask me anything about Pokemon competitive analysis, team building, or strategies!</p>
              </div>
            )}
            
            {messages.map((message, index) => (
              <div key={index} className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`max-w-3xl p-4 rounded-lg ${
                  message.type === 'user' 
                    ? 'bg-blue-500 text-white' 
                    : message.type === 'error'
                    ? 'bg-red-100 text-red-700 border border-red-200'
                    : 'bg-gray-100 text-gray-800'
                }`}>
                  {message.type === 'error' && (
                    <div className="flex items-center mb-2">
                      <AlertCircle size={16} className="mr-2" />
                      <span className="font-semibold">Error</span>
                    </div>
                  )}
                  <div 
                    className="prose prose-sm max-w-none"
                    dangerouslySetInnerHTML={{ __html: formatMessage(message.content) }}
                  />
                  <div className="text-xs opacity-70 mt-2">
                    {new Date(message.timestamp).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input */}
        <form onSubmit={handleSubmit} className="bg-white rounded-lg shadow-lg p-4">
          <div className="flex space-x-4">
            <input
              type="text"
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Ask me about Pokemon competitive analysis, team building, comparisons..."
              className="flex-1 p-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={isLoading || !inputValue.trim()}
              className="px-6 py-3 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
            >
              {isLoading ? (
                <Loader2 className="animate-spin" size={20} />
              ) : (
                <Send size={20} />
              )}
            </button>
          </div>
        </form>

        {/* Footer */}
        <div className="text-center mt-8 text-gray-500 text-sm">
          <p>Powered by OpenAI GPT-4 and PokéAPI • Real-time competitive analysis</p>
        </div>
      </div>
    </div>
  );
}

export default App;

