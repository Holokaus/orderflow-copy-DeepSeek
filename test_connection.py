#!/usr/bin/env python3
"""
Quick connectivity test for Binance API
Tests REST API and DNS resolution
"""
import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import socket
import time

load_dotenv()

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from data.exchange_connector import ExchangeConnector, ExchangeConfig


async def test_dns():
    """Test DNS resolution"""
    print("=" * 60)
    print("Testing DNS Resolution")
    print("=" * 60)
    
    try:
        ip = socket.gethostbyname("api.binance.com")
        print(f"✓ DNS resolved: api.binance.com → {ip}")
        return True
    except socket.gaierror as e:
        print(f"✗ DNS resolution failed: {e}")
        print("  → Check your internet connection or firewall")
        return False


async def test_rest_api():
    """Test REST API connection"""
    print("\n" + "=" * 60)
    print("Testing REST API Connection (Anonymous)")
    print("=" * 60)
    
    api_key = os.getenv("BINANCE_API_KEY", "")
    api_secret = os.getenv("BINANCE_API_SECRET", "")
    use_testnet = True  # Use testnet by default
    
    if not api_key:
        print("⚠️  No API key in .env - testing with ANONYMOUS mode")
        print("    (This tests connectivity but can't verify your credentials)")
    
    try:
        # Test with empty credentials first (anonymous mode)
        exchange = ExchangeConnector(ExchangeConfig(
            exchange_id="binance",
            api_key="",  # Empty = anonymous
            api_secret="",  # Empty = anonymous
            testnet=use_testnet
        ))
        
        mode_str = "Testnet" if use_testnet else "Production"
        print(f"Attempting to connect ({mode_str}, anonymous mode)...")
        start = time.time()
        await exchange.connect()
        elapsed = time.time() - start
        
        print(f"✓ Connected successfully ({elapsed:.2f}s)")
        await exchange.disconnect()
        
        # If anonymous works, test with credentials
        if api_key and api_secret:
            print("\n✓ Anonymous connection works. Now testing WITH credentials...")
            exchange2 = ExchangeConnector(ExchangeConfig(
                exchange_id="binance",
                api_key=api_key,
                api_secret=api_secret,
                testnet=use_testnet
            ))
            print("Attempting to connect with API credentials...")
            start = time.time()
            await exchange2.connect()
            elapsed = time.time() - start
            print(f"✓ Authenticated connection works! ({elapsed:.2f}s)")
            await exchange2.disconnect()
        
        return True
        
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        error_msg = str(e).lower()
        
        if "timeout" in error_msg:
            print("\n  TIMEOUT ISSUE:")
            print("  - Binance API is slow or unreachable")
            print("  - Check your internet connection")
            print("  - Try again in a few moments")
        elif "geo" in error_msg or "restricted" in error_msg:
            print("\n  GEO-RESTRICTION:")
            print("  - Your location is blocked by Binance")
            print("  - Use a VPN to a different country")
        elif "401" in error_msg or "invalid api" in error_msg.lower():
            print("\n  AUTHENTICATION ERROR:")
            print("  - Your API credentials are invalid or expired")
            print("  - Option 1: Get FRESH credentials from demo.binance.com")
            print("  - Option 2: Use PRODUCTION credentials from binance.com")
            print("             (Paper trading is simulated, 100% safe)")
        else:
            print(f"\n  Other error: {error_msg}")
        
        return False


async def test_websocket_url():
    """Test WebSocket connectivity"""
    print("\n" + "=" * 60)
    print("Testing WebSocket URL")
    print("=" * 60)
    
    ws_url = "wss://stream.binance.com:9443"
    
    try:
        import websockets
        print(f"Testing connection to {ws_url}...")
        
        async with websockets.connect(ws_url, ping_interval=None) as ws:
            print(f"✓ WebSocket connected successfully")
            return True
    except ImportError:
        print("⚠️  websockets module not installed (optional)")
        return None
    except Exception as e:
        print(f"✗ WebSocket connection failed: {e}")
        return False


async def main():
    print("\n" + "=" * 60)
    print("BINANCE CONNECTIVITY DIAGNOSTIC")
    print("=" * 60 + "\n")
    
    results = {
        "DNS": await test_dns(),
        "REST API": await test_rest_api(),
        "WebSocket": await test_websocket_url(),
    }
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for test_name, result in results.items():
        if result is None:
            status = "⊘ SKIPPED"
        elif result:
            status = "✓ PASS"
        else:
            status = "✗ FAIL"
        print(f"{test_name:20s} {status}")
    
    print("=" * 60)
    
    if all(r for r in results.values() if r is not None):
        print("\n✓ All tests passed! Ready to run paper trading.")
        return 0
    else:
        print("\n✗ Some tests failed. Check the details above.")
        print("\nTroubleshooting steps:")
        print("1. Check your internet connection")
        print("2. Disable any VPN/proxy temporarily")
        print("3. Check if Binance is accessible from your country")
        print("4. Try again in a few moments")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
