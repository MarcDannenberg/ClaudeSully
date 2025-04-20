import time
from typing import List, Dict, Optional
from datetime import datetime


class MarketMemory:
    """
    A system to track and match buyers and sellers in a marketplace.
    
    Stores information about buyers and sellers including their items,
    categories, budgets/prices, and user identifiers.
    """
    
    def __init__(self):
        """Initialize empty lists to store buyer and seller information."""
        self.buyers = []
        self.sellers = []

    def log_buyer(self, item: str, category: str, budget: str, user: str, permission: bool) -> bool:
        """
        Log a buyer's interest in an item.
        
        Args:
            item: The item the buyer is interested in
            category: The category of the item
            budget: The buyer's budget for the item
            user: Identifier for the buyer
            permission: Whether the buyer has given permission to log their interest
            
        Returns:
            bool: True if the buyer was successfully logged, False otherwise
        """
        if not permission:
            return False
            
        self.buyers.append({
            "type": "buyer",
            "item": item,
            "category": category,
            "budget": budget,
            "user": user,
            "timestamp": time.time()
        })
        return True

    def log_seller(self, item: str, category: str, price: str, user: str, permission: bool) -> bool:
        """
        Log a seller's listing of an item.
        
        Args:
            item: The item the seller is listing
            category: The category of the item
            price: The seller's asking price for the item
            user: Identifier for the seller
            permission: Whether the seller has given permission to log their listing
            
        Returns:
            bool: True if the seller was successfully logged, False otherwise
        """
        if not permission:
            return False
            
        self.sellers.append({
            "type": "seller",
            "item": item,
            "category": category,
            "price": price,
            "user": user,
            "timestamp": time.time()
        })
        return True

    def match_listings(self) -> List[Dict]:
        """
        Match buyers with sellers based on item categories and items.
        
        Returns:
            List[Dict]: A list of matches sorted by match score (highest first)
        """
        matches = []
        
        for buyer in self.buyers:
            buyer_category = buyer["category"].lower()
            buyer_item = buyer["item"].lower()
            
            for seller in self.sellers:
                seller_category = seller["category"].lower()
                seller_item = seller["item"].lower()
                
                # Only match if categories match
                if buyer_category == seller_category:
                    # Calculate match score - perfect match if buyer's item is in seller's item
                    match_score = 1.0 if buyer_item in seller_item else 0.5
                    
                    matches.append({
                        "buyer": buyer,
                        "seller": seller,
                        "score": match_score,
                        "match_time": datetime.now().isoformat()
                    })
        
        # Sort matches by score (highest first)
        return sorted(matches, key=lambda x: x["score"], reverse=True)
    
    def get_recent_buyers(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent buyers.
        
        Args:
            limit: Maximum number of buyers to return
            
        Returns:
            List[Dict]: List of recent buyers, sorted by timestamp (newest first)
        """
        return sorted(self.buyers, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def get_recent_sellers(self, limit: int = 10) -> List[Dict]:
        """
        Get the most recent sellers.
        
        Args:
            limit: Maximum number of sellers to return
            
        Returns:
            List[Dict]: List of recent sellers, sorted by timestamp (newest first)
        """
        return sorted(self.sellers, key=lambda x: x["timestamp"], reverse=True)[:limit]
    
    def clear_old_listings(self, max_age_days: int = 30) -> int:
        """
        Remove listings older than a specified number of days.
        
        Args:
            max_age_days: Maximum age of listings in days
            
        Returns:
            int: Number of listings removed
        """
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 60 * 60
        
        original_buyer_count = len(self.buyers)
        original_seller_count = len(self.sellers)
        
        # Remove old buyers
        self.buyers = [buyer for buyer in self.buyers 
                      if current_time - buyer["timestamp"] <= max_age_seconds]
        
        # Remove old sellers
        self.sellers = [seller for seller in self.sellers 
                       if current_time - seller["timestamp"] <= max_age_seconds]
        
        # Return number of removed listings
        return (original_buyer_count - len(self.buyers)) + (original_seller_count - len(self.sellers))


# Create a singleton instance for global use
market = MarketMemory()