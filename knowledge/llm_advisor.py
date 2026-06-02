"""
LLM Advisor
Interface to external LLM APIs for knowledge extraction and guidance.
Supports 9 LLM providers: OpenAI, Gemini, Anthropic, OpenRouter, GitHub Models, xAI, Z.ai, Groq, Ollama.
"""

import json
import hashlib
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from functools import lru_cache
import os

from loguru import logger

# Import LLMProvider from config as single source of truth
from config.settings import LLMProvider, settings

# API clients - import conditionally
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

# Re-export LLMProvider for backward compatibility with main.py
__all__ = ['LLMAdvisor', 'LLMProvider', 'LLMResponse', 'ResponseCache', 
           'ABSORPTION_KNOWLEDGE', 'DELTA_DIVERGENCE_KNOWLEDGE', 
           'LIQUIDITY_SWEEP_KNOWLEDGE', 'STACKED_IMBALANCE_KNOWLEDGE']


@dataclass
class LLMResponse:
    """Structured response from LLM"""
    content: str
    parsed: Optional[Dict] = None
    provider: str = ""
    model: str = ""
    tokens_used: int = 0
    cached: bool = False


class ResponseCache:
    """Simple in-memory cache for LLM responses (includes provider+model in hash to prevent collisions)"""
    
    def __init__(self, ttl_seconds: int = 3600):
        self.cache: Dict[str, tuple] = {}  # hash -> (response, timestamp)
        self.ttl = timedelta(seconds=ttl_seconds)
    
    def _hash_prompt(self, prompt: str, provider: str = "", model: str = "") -> str:
        """Hash prompt with provider+model to prevent cross-provider collisions"""
        combined = f"{provider}:{model}:{prompt}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def get(self, prompt: str, provider: str = "", model: str = "") -> Optional[str]:
        key = self._hash_prompt(prompt, provider, model)
        if key in self.cache:
            response, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.ttl:
                return response
            else:
                del self.cache[key]
        return None
    
    def set(self, prompt: str, response: str, provider: str = "", model: str = ""):
        key = self._hash_prompt(prompt, provider, model)
        self.cache[key] = (response, datetime.now())


class LLMAdvisor:
    """
    Interface to LLM APIs for trading knowledge extraction.
    
    Supports 9 providers:
    - OPENAI: OpenAI GPT models
    - GEMINI: Google Gemini
    - ANTHROPIC: Anthropic Claude
    - OPENROUTER: OpenRouter (OpenAI-compatible)
    - GITHUB_MODELS: GitHub Models (OpenAI-compatible)
    - XAI: xAI Grok (OpenAI-compatible)
    - ZAI: Z.ai/Zhipu (OpenAI-compatible)
    - GROQ: Groq (OpenAI-compatible)
    - OLLAMA: Local Ollama (OpenAI-compatible, no API key)
    
    Uses LLMs for:
    1. Feature engineering guidance
    2. Parameter range suggestions
    3. Strategy rule extraction
    4. Pattern interpretation
    5. Weak labeling of setups
    """
    
    # Provider configuration: compatible providers mapping
    _PROVIDER_CONFIG = {
        # OpenAI-compatible providers
        LLMProvider.OPENAI: {
            "type": "openai",
            "base_url": None,
            "default_model": "gpt-4o",
        },
        LLMProvider.OPENROUTER: {
            "type": "openai",
            "base_url": "https://openrouter.ai/api/v1",
            "default_model": "anthropic/claude-3.5-sonnet",
        },
        LLMProvider.GITHUB_MODELS: {
            "type": "openai",
            "base_url": "https://models.inference.ai.azure.com",
            "default_model": "gpt-4o",
        },
        LLMProvider.XAI: {
            "type": "openai",
            "base_url": "https://api.x.ai/v1",
            "default_model": "grok-2-latest",
        },
        LLMProvider.ZAI: {
            "type": "openai",
            "base_url": "https://api.z.ai/api/paas/v4",
            "default_model": "glm-5",
        },
        LLMProvider.GROQ: {
            "type": "openai",
            "base_url": "https://api.groq.com/openai/v1",
            "default_model": "llama-3.3-70b-versatile",
        },
        LLMProvider.OLLAMA: {
            "type": "openai",
            "base_url": "http://localhost:11434/v1",
            "default_model": "llama3",
        },
        # Native SDK providers
        LLMProvider.GEMINI: {
            "type": "gemini",
            "default_model": "gemini-2.0-flash",
        },
        LLMProvider.ANTHROPIC: {
            "type": "anthropic",
            "default_model": "claude-sonnet-4-20250514",
        },
    }
    
    def __init__(
        self,
        provider: Union[LLMProvider, str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        cache_responses: bool = True,
        base_url: Optional[str] = None,
    ):
        """
        Initialize LLM Advisor with specified provider.
        
        Args:
            provider: LLMProvider enum or string value
            model: Model name (optional, uses provider default if None)
            api_key: API key (optional, reads from settings/env vars if None)
            cache_responses: Enable response caching
            base_url: Custom base URL for OpenAI-compatible providers
        """
        # Handle provider input
        if provider is None:
            provider = settings.llm.provider
        elif isinstance(provider, str):
            provider = LLMProvider(provider)
        
        self.provider = provider
        self.cache = ResponseCache(settings.llm.cache_ttl_seconds) if cache_responses else None
        
        # Get provider config
        if provider not in self._PROVIDER_CONFIG:
            raise ValueError(f"Unknown provider: {provider}")
        
        config = self._PROVIDER_CONFIG[provider]
        self.model = model or settings.llm.model or config["default_model"]
        
        # Resolve API key with priority: arg > settings > env var
        resolved_key = self._resolve_api_key(provider, api_key)
        
        # Initialize client based on provider type
        if config["type"] == "openai":
            self._init_openai_compatible(provider, config, resolved_key, base_url)
        elif config["type"] == "gemini":
            self._init_gemini(resolved_key)
        elif config["type"] == "anthropic":
            self._init_anthropic(resolved_key)
    
    @classmethod
    def create_with_failover(cls, llm_config: 'LLMConfig') -> Optional['LLMAdvisor']:
        """
        Factory method that attempts to initialize LLM advisor with automatic failover.
        
        Tries providers in order from llm_config.get_failover_chain() until one succeeds.
        Performs a test API call to verify each provider works before returning.
        
        Args:
            llm_config: LLMConfig instance with failover configuration
            
        Returns:
            Initialized LLMAdvisor instance, or None if all providers fail
            
        Example:
            advisor = LLMAdvisor.create_with_failover(settings.llm)
            if advisor is None:
                logger.warning("No LLM available, using fallback constants")
        """
        if not llm_config.enable_failover:
            # Simple initialization without failover
            try:
                api_key = llm_config.get_api_key(llm_config.provider)
                base_url = llm_config.get_base_url(llm_config.provider)
                return cls(
                    provider=llm_config.provider,
                    model=llm_config.model,
                    api_key=api_key,
                    cache_responses=llm_config.cache_responses,
                    base_url=base_url
                )
            except Exception as e:
                logger.error(f"LLM initialization failed: {e}")
                return None
        
        # Failover mode: try each provider in chain
        chain = llm_config.get_failover_chain()
        
        for provider, api_key, base_url in chain:
            try:
                logger.info(f"Attempting LLM provider: {provider.value}")
                
                advisor = cls(
                    provider=provider,
                    model=llm_config.model,
                    api_key=api_key,
                    cache_responses=llm_config.cache_responses,
                    base_url=base_url
                )
                
                # Verify with test call
                test_response = advisor._call_llm("Test connection")
                
                if test_response and test_response.content:
                    logger.info(f"LLM initialized successfully: {provider.value}")
                    return advisor
                else:
                    raise ValueError("Empty response from test call")
                    
            except Exception as e:
                error_msg = str(e)
                # Truncate long error messages (e.g., Gemini quota errors)
                if len(error_msg) > 100:
                    error_msg = error_msg[:100] + "..."
                logger.warning(f"{provider.value} failed: {error_msg}")
                continue  # Try next provider
        
        logger.error("All LLM providers exhausted, no LLM available")
        return None
    
    def _resolve_api_key(self, provider: LLMProvider, explicit_key: Optional[str]) -> Optional[str]:
        """Resolve API key with priority: explicit arg > settings > env var"""
        if explicit_key:
            return explicit_key
        
        # Try settings first
        llm_config = settings.llm
        key_map = {
            LLMProvider.OPENAI: llm_config.openai_api_key,
            LLMProvider.GEMINI: llm_config.gemini_api_key,
            LLMProvider.ANTHROPIC: llm_config.anthropic_api_key,
            LLMProvider.OPENROUTER: llm_config.openrouter_api_key,
            LLMProvider.GITHUB_MODELS: llm_config.github_token,
            LLMProvider.XAI: llm_config.xai_api_key,
            LLMProvider.ZAI: llm_config.zai_api_key,
            LLMProvider.GROQ: llm_config.groq_api_key,
            LLMProvider.OLLAMA: "ollama",  # Dummy value - no auth needed
        }
        
        if provider in key_map:
            key = key_map[provider]
            if key:
                return key
        
        # Fallback to env var
        env_map = {
            LLMProvider.OPENAI: "OPENAI_API_KEY",
            LLMProvider.GEMINI: "GEMINI_API_KEY",
            LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
            LLMProvider.OPENROUTER: "OPENROUTER_API_KEY",
            LLMProvider.GITHUB_MODELS: "GITHUB_TOKEN",
            LLMProvider.XAI: "XAI_API_KEY",
            LLMProvider.ZAI: "ZAI_API_KEY",
            LLMProvider.GROQ: "GROQ_API_KEY",
            LLMProvider.OLLAMA: None,
        }
        
        env_var = env_map.get(provider)
        if env_var:
            return os.getenv(env_var)
        
        if provider == LLMProvider.OLLAMA:
            return "ollama"  # Dummy value
        
        return None
    
    def _init_openai_compatible(
        self,
        provider: LLMProvider,
        config: Dict,
        api_key: Optional[str],
        custom_base_url: Optional[str]
    ):
        """Initialize OpenAI-compatible provider"""
        if not HAS_OPENAI:
            raise ImportError(
                f"openai package not installed. Install with: pip install openai"
            )
        
        # Use custom base URL if provided, otherwise use config
        base_url = custom_base_url or config.get("base_url")
        
        # Prepare initialization kwargs
        init_kwargs = {"api_key": api_key or "dummy"}  # Dummy key if none provided
        if base_url:
            init_kwargs["base_url"] = base_url
        
        self.client = openai.OpenAI(**init_kwargs)
        
        # Store provider-specific headers for OpenRouter
        self.provider_headers = {}
        if provider == LLMProvider.OPENROUTER:
            if settings.llm.openrouter_site_url:
                self.provider_headers["HTTP-Referer"] = settings.llm.openrouter_site_url
            if settings.llm.openrouter_app_name:
                self.provider_headers["X-Title"] = settings.llm.openrouter_app_name
    
    def _init_gemini(self, api_key: Optional[str]):
        """Initialize Google Gemini"""
        if not HAS_GEMINI:
            raise ImportError(
                f"google-generativeai package not installed. Install with: pip install google-generativeai"
            )
        
        if not api_key:
            raise ValueError(
                "Gemini API key not found. Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=api_key)
        
        # Store safety settings for later use
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
        
        # Create Gemini client
        self.client = genai.GenerativeModel(self.model)
    
    def _init_anthropic(self, api_key: Optional[str]):
        """Initialize Anthropic Claude"""
        if not HAS_ANTHROPIC:
            raise ImportError(
                f"anthropic package not installed. Install with: pip install anthropic"
            )
        
        if not api_key:
            raise ValueError(
                "Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def _call_llm(self, prompt: str, system_prompt: Optional[str] = None) -> LLMResponse:
        """Make API call to LLM"""
        
        # Check cache
        if self.cache:
            cached = self.cache.get(prompt, self.provider.value, self.model)
            if cached:
                logger.debug(f"Using cached LLM response (provider={self.provider.value}, model={self.model})")
                return LLMResponse(content=cached, cached=True, provider=self.provider.value, model=self.model)
        
        try:
            # Route to appropriate provider
            config = self._PROVIDER_CONFIG[self.provider]
            
            if config["type"] == "openai":
                # OpenAI-compatible provider
                content, tokens = self._call_openai_compatible(prompt, system_prompt)
            elif config["type"] == "gemini":
                # Google Gemini
                content, tokens = self._call_gemini(prompt, system_prompt)
            elif config["type"] == "anthropic":
                # Anthropic Claude
                content, tokens = self._call_anthropic(prompt, system_prompt)
            else:
                raise ValueError(f"Unknown provider type: {config['type']}")
            
            # Cache response
            if self.cache:
                self.cache.set(prompt, content, self.provider.value, self.model)
            
            return LLMResponse(
                content=content,
                provider=self.provider.value,
                model=self.model,
                tokens_used=tokens
            )
            
        except Exception as e:
            logger.error(f"LLM API call failed ({self.provider.value}): {e}")
            raise
    
    def _call_openai_compatible(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        """Call OpenAI-compatible API"""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # Prepare kwargs for API call
        call_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.3,
        }
        
        # Add provider-specific headers for OpenRouter
        if self.provider == LLMProvider.OPENROUTER and self.provider_headers:
            call_kwargs["extra_headers"] = self.provider_headers
        
        response = self.client.chat.completions.create(**call_kwargs)
        content = response.choices[0].message.content
        tokens = getattr(response.usage, 'total_tokens', 0)
        
        return content, tokens
    
    def _call_gemini(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        """Call Google Gemini API"""
        full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        response = self.client.generate_content(
            full_prompt,
            safety_settings=self.safety_settings
        )
        content = response.text
        tokens = 0  # Gemini doesn't always return token count
        
        return content, tokens
    
    def _call_anthropic(self, prompt: str, system_prompt: Optional[str]) -> tuple:
        """Call Anthropic Claude API"""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=4096,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )
        content = response.content[0].text
        tokens = response.usage.input_tokens + response.usage.output_tokens
        
        return content, tokens
    
    def _parse_json_response(self, response: LLMResponse) -> Dict:
        """Extract JSON from LLM response"""
        content = response.content
        
        # Try to find JSON block
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            content = content[start:end].strip()
        
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from LLM response")
            return {}
    
    # ==================== KNOWLEDGE EXTRACTION METHODS ====================
    
    def get_parameter_ranges(self, pattern_name: str) -> Dict[str, Dict[str, float]]:
        """
        Get suggested parameter ranges for a specific pattern.
        
        Returns dict like:
        {
            "volume_threshold": {"min": 1.5, "max": 4.0, "default": 2.0},
            "price_threshold": {"min": 0.0005, "max": 0.002, "default": 0.001},
            ...
        }
        """
        system_prompt = """You are an expert quantitative trader specializing in order flow analysis.
        Provide parameter ranges based on your domain knowledge of market microstructure.
        Be specific and practical. Return valid JSON only."""
        
        prompt = f"""For detecting "{pattern_name}" patterns in BTCUSD order flow data, 
        what are the recommended parameter ranges?
        
        Consider:
        - Typical values used by professional traders
        - Ranges that work across different market conditions
        - Conservative defaults that avoid false positives
        
        Return JSON in this exact format:
        {{
            "parameter_name": {{
                "min": <minimum_value>,
                "max": <maximum_value>,
                "default": <recommended_default>,
                "description": "<what this parameter controls>"
            }},
            ...
        }}
        
        Include all relevant parameters for {pattern_name} detection."""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def get_feature_importance(self, strategy_type: str) -> Dict[str, float]:
        """
        Get feature importance weights for a strategy type.
        
        Returns dict like:
        {
            "delta_imbalance": 0.8,
            "book_pressure": 0.6,
            ...
        }
        """
        system_prompt = """You are an expert in order flow trading strategies.
        Provide feature importance scores based on domain expertise.
        Scores should be 0-1 where 1 is most important."""
        
        prompt = f"""For a "{strategy_type}" order flow trading strategy on BTCUSD,
        rate the importance of each feature (0-1 scale).
        
        Consider these feature categories:
        1. Order book features (depth, imbalance, spread, microprice)
        2. Trade flow features (delta, volume, aggressor ratio)
        3. Volume profile features (POC, value area, VWAP)
        4. Footprint features (delta at price, stacked imbalances)
        5. Pattern detection (absorption, sweeps, icebergs)
        
        Return JSON with feature names as keys and importance scores as values.
        Include at least 20 features."""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def get_strategy_rules(self, strategy_name: str) -> Dict[str, Any]:
        """
        Extract explicit trading rules for a strategy.
        
        Returns structured rules that can be encoded into logic.
        """
        system_prompt = """You are an expert order flow trader.
        Provide specific, actionable trading rules that can be implemented in code.
        Be precise about conditions, thresholds, and logic."""
        
        prompt = f"""Define the trading rules for a "{strategy_name}" strategy.
        
        Provide:
        1. Entry conditions (specific feature thresholds)
        2. Exit conditions (stop loss, take profit logic)
        3. Position sizing rules
        4. Filter conditions (when NOT to trade)
        5. Regime-specific adjustments
        
        Return JSON in this format:
        {{
            "entry_conditions": [
                {{"feature": "...", "operator": ">", "threshold": ..., "weight": ...}},
                ...
            ],
            "exit_conditions": {{
                "stop_loss": {{"type": "...", "value": ...}},
                "take_profit": {{"type": "...", "value": ...}}
            }},
            "filters": [...],
            "position_sizing": {{...}},
            "regime_adjustments": {{...}}
        }}"""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def label_setup(self, features: Dict[str, float], context: str = "") -> Dict[str, Any]:
        """
        Get LLM to label a trading setup for weak supervision.
        
        Returns quality score and reasoning.
        """
        system_prompt = """You are an expert order flow analyst evaluating trading setups.
        Rate the quality of this setup based on the provided features.
        Be specific about why this is or isn't a good setup."""
        
        # Format features for prompt
        feature_str = "\n".join([f"- {k}: {v:.4f}" for k, v in features.items()])
        
        prompt = f"""Evaluate this order flow setup for BTCUSD:
        
        Features:
        {feature_str}
        
        Context: {context if context else "No additional context"}
        
        Return JSON:
        {{
            "quality_score": <0-100>,
            "direction": "<LONG/SHORT/NEUTRAL>",
            "confidence": <0-100>,
            "primary_signal": "<main reason for or against>",
            "supporting_signals": ["<additional positive factors>"],
            "warning_signals": ["<concerns or red flags>"],
            "suggested_stop_distance_pct": <0.001-0.01>,
            "suggested_target_distance_pct": <0.001-0.02>
        }}"""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def get_initial_parameters(self, strategy_type: str) -> Dict[str, float]:
        """
        Get initial/default parameter values for Optuna warm start.
        
        Returns a single set of "best guess" parameters.
        """
        system_prompt = """You are an expert quantitative trader.
        Provide your best estimate for initial parameter values.
        These will be used as starting points for optimization."""
        
        prompt = f"""For a "{strategy_type}" order flow trading strategy on BTCUSD,
        provide your best initial parameter values.
        
        Consider:
        - BTCUSD typical volatility and spread
        - Common institutional order flow patterns
        - Conservative values that work across conditions
        
        Return JSON with parameter names and values:
        {{
            "absorption_volume_mult": <value>,
            "absorption_price_threshold": <value>,
            "imbalance_ratio_threshold": <value>,
            "delta_threshold": <value>,
            "lookback_seconds": <value>,
            "min_holding_seconds": <value>,
            "stop_loss_atr_mult": <value>,
            "take_profit_atr_mult": <value>,
            ...include all relevant parameters...
        }}"""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)
    
    def analyze_performance(
        self, 
        metrics: Dict[str, float],
        recent_trades: List[Dict]
    ) -> Dict[str, Any]:
        """
        Get LLM analysis of strategy performance and suggestions.
        """
        system_prompt = """You are a trading performance analyst.
        Analyze the provided metrics and trades to identify issues and improvements."""
        
        metrics_str = "\n".join([f"- {k}: {v:.4f}" for k, v in metrics.items()])
        trades_str = json.dumps(recent_trades[:20], indent=2)
        
        prompt = f"""Analyze this trading strategy performance:
        
        Metrics:
        {metrics_str}
        
        Recent trades (sample):
        {trades_str}
        
        Provide:
        1. Overall assessment
        2. Key issues identified
        3. Specific parameter adjustments to consider
        4. Market conditions where strategy underperforms
        
        Return JSON:
        {{
            "assessment": "<overall evaluation>",
            "grade": "<A/B/C/D/F>",
            "issues": ["<issue 1>", ...],
            "suggested_adjustments": {{
                "<parameter>": {{"current_implied": ..., "suggested": ..., "reason": "..."}}
            }},
            "avoid_conditions": ["<condition where strategy fails>", ...]
        }}"""
        
        response = self._call_llm(prompt, system_prompt)
        return self._parse_json_response(response)


# ==================== DOMAIN KNOWLEDGE CONSTANTS ====================
# Pre-extracted knowledge to avoid repeated API calls

ABSORPTION_KNOWLEDGE = {
    "parameter_ranges": {
        "volume_multiplier": {"min": 1.5, "max": 5.0, "default": 2.5},
        "price_threshold_pct": {"min": 0.0003, "max": 0.002, "default": 0.001},
        "min_duration_seconds": {"min": 1.0, "max": 10.0, "default": 3.0},
        "strength_threshold": {"min": 0.4, "max": 0.8, "default": 0.6},
    },
    "rules": {
        "entry": "Volume > multiplier * avg AND price_change < threshold AND duration > min_duration",
        "confirmation": "Look for failed auction (price returns to absorption level)",
        "invalidation": "Price breaks through absorption level with increasing delta"
    }
}

DELTA_DIVERGENCE_KNOWLEDGE = {
    "parameter_ranges": {
        "lookback_bars": {"min": 3, "max": 20, "default": 10},
        "price_threshold_pct": {"min": 0.001, "max": 0.01, "default": 0.003},
        "delta_threshold_pct": {"min": 0.1, "max": 0.5, "default": 0.25},
    },
    "rules": {
        "bullish_divergence": "Price makes lower low BUT delta makes higher low",
        "bearish_divergence": "Price makes higher high BUT delta makes lower high",
        "confirmation": "Wait for price to cross back above/below trigger level"
    }
}

LIQUIDITY_SWEEP_KNOWLEDGE = {
    "parameter_ranges": {
        "sweep_speed_max_seconds": {"min": 1.0, "max": 10.0, "default": 5.0},
        "sweep_depth_pct": {"min": 0.001, "max": 0.01, "default": 0.003},
        "reversal_threshold_pct": {"min": 0.3, "max": 0.8, "default": 0.5},
        "volume_spike_multiplier": {"min": 2.0, "max": 10.0, "default": 4.0},
    },
    "rules": {
        "identification": "Fast move through liquidity zone followed by reversal",
        "entry": "Enter on reversal confirmation (price reclaims key level)",
        "stop": "Place stop beyond the sweep extreme",
        "target": "Target opposite liquidity zone or POC"
    }
}

STACKED_IMBALANCE_KNOWLEDGE = {
    "parameter_ranges": {
        "imbalance_ratio": {"min": 2.0, "max": 5.0, "default": 3.0},
        "min_stack_levels": {"min": 2, "max": 5, "default": 3},
        "volume_threshold": {"min": 0.5, "max": 2.0, "default": 1.0},
    },
    "rules": {
        "identification": "3+ consecutive price levels with same-side imbalance",
        "interpretation": "Indicates strong directional conviction",
        "entry": "Trade in direction of imbalance on pullback to stack",
        "invalidation": "Stack levels are traded through with opposing delta"
    }
}