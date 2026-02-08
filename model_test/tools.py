PRODUCTS = {
    "iPhone": {"name": "iPhone", "price": 999.99, "category": "electronics"},
    "iPhone 15": {"name": "iPhone 15", "price": 1099.99, "category": "electronics"},
    "Samsung Galaxy": {"name": "Samsung Galaxy", "price": 899.99, "category": "electronics"},
    "Samsung Galaxy S24": {"name": "Samsung Galaxy S24", "price": 999.99, "category": "electronics"},
    "Laptop": {"name": "Laptop", "price": 1299.99, "category": "electronics"},
    "Headphones": {"name": "Headphones", "price": 149.99, "category": "electronics"},
    "Wireless Headphones": {"name": "Wireless Headphones", "price": 199.99, "category": "electronics"},
    "Bluetooth Headphones": {"name": "Bluetooth Headphones", "price": 179.99, "category": "electronics"},
    "Sony Headphones": {"name": "Sony Headphones", "price": 249.99, "category": "electronics"},
    "Keyboard": {"name": "Mechanical Keyboard", "price": 149.99, "category": "electronics"},
    "Mouse": {"name": "Gaming Mouse", "price": 79.99, "category": "electronics"},
    "Programming Book": {"name": "Programming Book", "price": 49.99, "category": "books"},
    "Science Fiction Novel": {"name": "Science Fiction Novel", "price": 24.99, "category": "books"},
    "Cookbook": {"name": "Cookbook", "price": 34.99, "category": "books"},
}


def search_products(query: str = "", category: str = "") -> list[dict]:
    """Search for products by query or category."""
    results = []
    query_lower = query.lower()

    for product in PRODUCTS.values():
        # If query is provided, search in both name and category
        if query:
            name_match = query_lower in product["name"].lower()
            category_match = query_lower in product["category"].lower()
            if not (name_match or category_match):
                continue

        # If category is explicitly specified, must match exactly
        if category and category.lower() != product["category"].lower():
            continue

        results.append(product)

    return results


class CartService:
    def __init__(self):
        self.items: dict[str, int] = {}
    
    def add_to_cart(self, product_name: str, quantity: int = 1) -> dict:
        """Add product to cart."""
        self.items[product_name] = self.items.get(product_name, 0) + quantity
        return {"success": True, "message": f"Added {quantity} {product_name} to cart"}
    
    def remove_from_cart(self, product_name: str) -> dict:
        """Remove product from cart."""
        if product_name in self.items:
            del self.items[product_name]
            return {"success": True, "message": f"Removed {product_name} from cart"}
        return {"success": False, "message": f"{product_name} not in cart"}
    
    def view_cart(self) -> dict:
        """View cart contents."""
        items = [{"product_name": name, "quantity": qty} for name, qty in self.items.items()]
        total = sum(PRODUCTS.get(name, {}).get("price", 0) * qty for name, qty in self.items.items())
        return {"items": items, "total": total}
    
    def checkout(self) -> dict:
        """Process checkout."""
        if not self.items:
            return {"success": False, "message": "Cart is empty"}
        total = sum(PRODUCTS.get(name, {}).get("price", 0) * qty for name, qty in self.items.items())
        self.items.clear()
        return {"success": True, "message": f"Checkout complete. Total: ${total:.2f}"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Search for products by query or category",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "category": {"type": "string", "description": "Product category"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_to_cart",
            "description": "Add a product to the shopping cart",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Name of the product"},
                    "quantity": {"type": "integer", "description": "Quantity to add", "default": 1},
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remove_from_cart",
            "description": "Remove a product from the shopping cart",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {"type": "string", "description": "Name of the product"},
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "view_cart",
            "description": "View the contents of the shopping cart",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "checkout",
            "description": "Process checkout and complete the purchase",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def execute_tool(tool_name: str, arguments: dict, cart: CartService) -> str:
    """Execute a tool and return the result."""
    if tool_name == "search_products":
        results = search_products(**arguments)
        return f"Found {len(results)} products: {[p['name'] for p in results]}"
    elif tool_name == "add_to_cart":
        result = cart.add_to_cart(**arguments)
        return result["message"]
    elif tool_name == "remove_from_cart":
        result = cart.remove_from_cart(**arguments)
        return result["message"]
    elif tool_name == "view_cart":
        result = cart.view_cart()
        return f"Cart: {result['items']}, Total: ${result['total']:.2f}"
    elif tool_name == "checkout":
        result = cart.checkout()
        return result["message"]
    return "Unknown tool"
