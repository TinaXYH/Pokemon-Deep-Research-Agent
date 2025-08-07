#!/usr/bin/env python3
"""
Flask Backend API for Pokemon Deep Research Agent
Provides REST endpoints for the frontend to interact with the research system
"""

import os
import sys
import json
import asyncio
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import logging

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))


from src.core.conversation_memory import ConversationMemoryManager
from src.core.specification_manager import SpecificationManager
from src.core.query_disambiguator import QueryDisambiguator
from src.tools.llm_client import LLMClient
from src.tools.pokeapi_client import PokéAPIClient

app = Flask(__name__)
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
conversation_memory = ConversationMemoryManager()
llm_client = LLMClient(os.getenv('OPENAI_API_KEY'))
pokeapi_client = PokéAPIClient()
spec_manager = SpecificationManager()
query_disambiguator = QueryDisambiguator(spec_manager)

class ResearchAgent:
    """Simplified research agent for API integration"""
    
    def __init__(self):
        self.conversation_memory = conversation_memory
        self.spec_manager = spec_manager
        self.query_disambiguator = query_disambiguator
        self.llm_client = llm_client
        self.pokeapi_client = pokeapi_client
    
    async def process_query(self, query, conversation_id="default"):
        """Process a research query and return structured results"""
        try:
            # Phase 1: Query Analysis
            yield {
                "phase": "query_analysis",
                "status": "active",
                "message": "Analyzing query intent and generating research specification",
                "timestamp": datetime.now().isoformat()
            }
            
            # Get conversation context using correct method
            context = self.conversation_memory.get_persistent_context(conversation_id)
            history = self.conversation_memory.get_conversation_history(conversation_id, max_turns=3)
            
            # If conversation doesn't exist, start a new one with the given ID
            if not history and conversation_id not in self.conversation_memory.active_conversations:
                # For default conversation, create it if it doesn't exist
                if conversation_id == "default":
                    from src.core.conversation_memory import ConversationContext
                    context_obj = ConversationContext(
                        conversation_id="default",
                        created_at=datetime.now(),
                        last_updated=datetime.now(),
                        turns=[],
                        persistent_context={}
                    )
                    self.conversation_memory.active_conversations["default"] = context_obj
                    self.conversation_memory._save_conversation("default")
                context = {}
                history = []
            
            # Build context for classification
            context_for_classification = {
                'previous_turns': [{'user': turn.user_input, 'agent': turn.agent_response} for turn in history],
                'pokemon_discussed': list(context.get('pokemon_discussed', set())),
                'turn_count': len(history)
            }
            
            # Classify query
            spec = await self.spec_manager.analyze_query_and_generate_spec(query, context_for_classification)
            
            yield {
                "phase": "query_analysis",
                "status": "completed",
                "data": {
                    "query_type": spec.query_type.value,
                    "pokemon_names": spec.fields.get('target_pokemon', []) if isinstance(spec.fields.get('target_pokemon'), list) else [spec.fields.get('target_pokemon', '')],
                    "format": spec.fields.get('format', 'OU')
                },
                "timestamp": datetime.now().isoformat()
            }
            
            # Phase 2: Specification Confirmation
            yield {
                "phase": "specification_confirmation",
                "status": "active",
                "message": "Confirming research parameters and requirements",
                "timestamp": datetime.now().isoformat()
            }
            
            # For API mode, auto-confirm the specification
            spec.confirmed = True
            
            yield {
                "phase": "specification_confirmation",
                "status": "completed",
                "data": spec,
                "timestamp": datetime.now().isoformat()
            }
            
            # Phase 3: Data Collection
            yield {
                "phase": "data_collection",
                "status": "active",
                "message": "Retrieving Pokemon data from PokéAPI and knowledge base",
                "timestamp": datetime.now().isoformat()
            }
            
            # Collect Pokemon data
            pokemon_data = {}
            pokemon_names = []
            
            # Extract pokemon names from spec
            if 'target_pokemon' in spec.fields and spec.fields['target_pokemon']:
                if isinstance(spec.fields['target_pokemon'], list):
                    pokemon_names = spec.fields['target_pokemon']
                else:
                    pokemon_names = [spec.fields['target_pokemon']]
            elif 'pokemon_list' in spec.fields and spec.fields['pokemon_list']:
                pokemon_names = spec.fields['pokemon_list']
            
            for pokemon_name in pokemon_names:
                try:
                    data = await self.pokeapi_client.get_pokemon(pokemon_name.lower())
                    pokemon_data[pokemon_name] = data.model_dump() if hasattr(data, 'model_dump') else data
                except Exception as e:
                    logger.warning(f"Failed to get data for {pokemon_name}: {e}")
            
            yield {
                "phase": "data_collection",
                "status": "completed",
                "data": {"pokemon_count": len(pokemon_data)},
                "timestamp": datetime.now().isoformat()
            }
            
            # Phase 4: Deep Analysis
            yield {
                "phase": "deep_analysis",
                "status": "active",
                "message": "Performing competitive analysis and strategic evaluation",
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate analysis
            analysis_prompt = self._build_analysis_prompt(spec, pokemon_data, context_for_classification)
            analysis_response = await self.llm_client.chat_completion([
                {"role": "user", "content": analysis_prompt}
            ])
            analysis = analysis_response['choices'][0]['message']['content']
            
            yield {
                "phase": "deep_analysis",
                "status": "completed",
                "data": {"analysis_length": len(analysis)},
                "timestamp": datetime.now().isoformat()
            }
            
            # Phase 5: Report Generation
            yield {
                "phase": "report_generation",
                "status": "active",
                "message": "Generating comprehensive research report",
                "timestamp": datetime.now().isoformat()
            }
            
            # Generate final report
            report_prompt = self._build_report_prompt(spec, pokemon_data, analysis, context_for_classification)
            report_response = await self.llm_client.chat_completion([
                {"role": "user", "content": report_prompt}
            ])
            final_report = report_response['choices'][0]['message']['content']
            
            # Update conversation memory
            self.conversation_memory.add_turn(
                conversation_id, 
                query, 
                final_report,
                pokemon_mentioned=pokemon_names,
                query_type=spec.query_type.value,
                context_used=context_for_classification
            )
            
            yield {
                "phase": "report_generation",
                "status": "completed",
                "data": {
                    "report": final_report,
                    "research_type": spec.query_type.value,
                    "pokemon": pokemon_names
                },
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            yield {
                "phase": "error",
                "status": "error",
                "message": f"Research failed: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
    
    def _build_analysis_prompt(self, spec, pokemon_data, context):
        """Build analysis prompt based on specification and data"""
        pokemon_names = []
        
        # Extract pokemon names from spec
        if 'target_pokemon' in spec.fields and spec.fields['target_pokemon']:
            if isinstance(spec.fields['target_pokemon'], list):
                pokemon_names = spec.fields['target_pokemon']
            else:
                pokemon_names = [spec.fields['target_pokemon']]
        elif 'pokemon_list' in spec.fields and spec.fields['pokemon_list']:
            pokemon_names = spec.fields['pokemon_list']
        
        prompt = f"""
        Analyze the following Pokemon research request:
        
        Research Type: {spec.query_type.value}
        Pokemon: {', '.join(pokemon_names)}
        Format: {spec.fields.get('format', 'OU')}
        
        Pokemon Data Available: {len(pokemon_data)} Pokemon
        
        Provide a detailed competitive analysis focusing on:
        - Statistical analysis and tier placement
        - Role analysis and team positioning
        - Strengths and weaknesses
        - Counter strategies
        - Team synergies
        
        Keep the analysis professional and comprehensive.
        """
        return prompt
    
    def _build_report_prompt(self, spec, pokemon_data, analysis, context):
        """Build final report prompt"""
        prompt = f"""
        Generate a comprehensive Pokemon research report based on:
        
        Research Specification: {spec.to_dict()}
        Analysis: {analysis}
        
        Format the report as a professional research document with:
        1. Executive Summary
        2. Research Specification
        3. Detailed Analysis
        4. Practical Recommendations
        5. Conclusion
        
        Use markdown formatting for better readability.
        """
        return prompt

# Initialize research agent
research_agent = ResearchAgent()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

@app.route('/api/research', methods=['POST'])
def research_endpoint():
    """Main research endpoint that streams results"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        conversation_id = data.get('conversation_id', 'default')
        
        if not query:
            return jsonify({"error": "Query is required"}), 400
        
        def generate():
            """Generator function for streaming response"""
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                async def run_research():
                    async for result in research_agent.process_query(query, conversation_id):
                        yield f"data: {json.dumps(result)}\n\n"
                
                # Run the async generator
                async_gen = run_research()
                while True:
                    try:
                        result = loop.run_until_complete(async_gen.__anext__())
                        yield result
                    except StopAsyncIteration:
                        break
                        
            except Exception as e:
                logger.error(f"Error in research stream: {e}")
                yield f"data: {json.dumps({'phase': 'error', 'status': 'error', 'message': str(e)})}\n\n"
            finally:
                loop.close()
        
        return Response(
            generate(),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'Access-Control-Allow-Origin': '*'
            }
        )
        
    except Exception as e:
        logger.error(f"Error in research endpoint: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation/<conversation_id>/history', methods=['GET'])
def get_conversation_history(conversation_id):
    """Get conversation history"""
    try:
        history = conversation_memory.get_conversation_history(conversation_id)
        # Convert to serializable format
        history_data = []
        for turn in history:
            history_data.append({
                "turn_id": turn.turn_id,
                "timestamp": turn.timestamp.isoformat(),
                "user_input": turn.user_input,
                "agent_response": turn.agent_response,
                "pokemon_mentioned": turn.pokemon_mentioned,
                "query_type": turn.query_type
            })
        return jsonify({"history": history_data})
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/conversation/<conversation_id>/clear', methods=['POST'])
def clear_conversation(conversation_id):
    """Clear conversation history"""
    try:
        conversation_memory.clear_conversation(conversation_id)
        return jsonify({"message": "Conversation cleared"})
    except Exception as e:
        logger.error(f"Error clearing conversation: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for required environment variables
    if not os.getenv('OPENAI_API_KEY'):
        logger.error("OPENAI_API_KEY environment variable is required")
        sys.exit(1)
    
    # Run the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)

