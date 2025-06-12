import json
from pathlib import Path
import torch
import numpy as np
from typing import Dict, List, Optional, Any
import torch_geometric
from torch_geometric.data import Data

from hive.ml.featurise.endgame_to_data import process_endgame
from hive.trajectory.game_dataloader import GameDataLoader


class PyGToVisJS:
    """
    Converts PyTorch Geometric Data objects to vis.js compatible JSON format.
    """

    def __init__(self):
        self.node_colors = ['#97C2FC', '#FFAB91', '#81C784', '#FFD54F', '#F48FB1', '#CE93D8']
        self.edge_colors = ['#848484', '#2B7CE9', '#FFA500', '#FF6347', '#32CD32', '#DA70D6']

    def convert_data_to_visjs(self,
                              data: Data,
                              node_features: Optional[List[str]] = None,
                              edge_features: Optional[List[str]] = None,
                              layout_method: str = 'physics',
                              node_size_feature: Optional[str] = None,
                              node_color_feature: Optional[str] = None,
                              edge_width_feature: Optional[str] = None) -> Dict[str, Any]:
        """
        Convert PyTorch Geometric Data object to vis.js format.

        Args:
            data: PyTorch Geometric Data object
            node_features: List of node feature names to include in labels
            edge_features: List of edge feature names to include in labels
            layout_method: Layout method ('physics', 'random', 'circular')
            node_size_feature: Feature name to use for node sizing
            node_color_feature: Feature name to use for node coloring
            edge_width_feature: Feature name to use for edge width

        Returns:
            Dictionary containing nodes and edges in vis.js format
        """

        # Extract basic graph structure
        num_nodes = data.num_nodes
        edge_index = data.edge_index.numpy() if isinstance(data.edge_index, torch.Tensor) else data.edge_index

        # Create nodes
        nodes = self._create_nodes(data, num_nodes, node_features, node_size_feature, node_color_feature)

        # Create edges
        edges = self._create_edges(data, edge_index, edge_features, edge_width_feature)

        # Create vis.js compatible structure
        visjs_data = {
            "nodes": nodes,
            "edges": edges,
            "options": self._create_options(layout_method)
        }

        return visjs_data

    def _create_nodes(self, data: Data, num_nodes: int, node_features: Optional[List[str]],
                      node_size_feature: Optional[str], node_color_feature: Optional[str]) -> List[Dict]:
        """Create nodes list for vis.js"""
        nodes = []

        for i in range(num_nodes):
            node = {
                "id": int(i),
                "label": f"Node {i}"
            }

            # Add node features to label if specified
            if node_features and hasattr(data, 'x') and data.x is not None:
                feature_values = []
                x = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x

                for j, feature_name in enumerate(node_features):
                    if j < x.shape[1]:
                        feature_values.append(f"{feature_name}: {float(x[i, j]):.3f}")

                if feature_values:
                    node["title"] = "\\n".join(feature_values)

            # Set node size based on feature
            if node_size_feature and hasattr(data, 'x') and data.x is not None:
                x = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x
                if node_features and node_size_feature in node_features:
                    feature_idx = node_features.index(node_size_feature)
                    if feature_idx < x.shape[1]:
                        # Normalize size between 10 and 50
                        feature_val = float(x[i, feature_idx])
                        x_min = float(np.min(x[:, feature_idx]))
                        x_max = float(np.max(x[:, feature_idx]))
                        normalized_size = 10 + (feature_val - x_min) / (x_max - x_min + 1e-8) * 40
                        node["size"] = int(normalized_size)

            # Set node color based on feature or class
            if node_color_feature:
                if hasattr(data, 'y') and data.y is not None and node_color_feature == 'class':
                    y = data.y.numpy() if isinstance(data.y, torch.Tensor) else data.y
                    class_idx = int(y[i]) if i < len(y) else 0
                    node["color"] = self.node_colors[class_idx % len(self.node_colors)]
                elif hasattr(data,
                             'x') and data.x is not None and node_features and node_color_feature in node_features:
                    x = data.x.numpy() if isinstance(data.x, torch.Tensor) else data.x
                    feature_idx = node_features.index(node_color_feature)
                    if feature_idx < x.shape[1]:
                        # Map feature value to color
                        feature_val = float(x[i, feature_idx])
                        x_min = float(np.min(x[:, feature_idx]))
                        x_max = float(np.max(x[:, feature_idx]))
                        color_idx = int((feature_val - x_min) / (x_max - x_min + 1e-8) * (len(self.node_colors) - 1))
                        node["color"] = self.node_colors[color_idx]
            else:
                # Default color based on node class if available
                if hasattr(data, 'y') and data.y is not None:
                    y = data.y.numpy() if isinstance(data.y, torch.Tensor) else data.y
                    class_idx = int(y[i]) if i < len(y) else 0
                    node["color"] = self.node_colors[class_idx % len(self.node_colors)]
                else:
                    node["color"] = self.node_colors[0]

            nodes.append(node)

        return nodes

    def _create_edges(self, data: Data, edge_index: np.ndarray, edge_features: Optional[List[str]],
                      edge_width_feature: Optional[str]) -> List[Dict]:
        """Create edges list for vis.js"""
        edges = []

        for i in range(edge_index.shape[1]):
            edge = {
                "id": int(i),
                "from": int(edge_index[0, i]),
                "to": int(edge_index[1, i])
            }

            # Add edge features to label if specified
            if edge_features and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                feature_values = []
                edge_attr = data.edge_attr.numpy() if isinstance(data.edge_attr, torch.Tensor) else data.edge_attr

                for j, feature_name in enumerate(edge_features):
                    if j < edge_attr.shape[1]:
                        feature_values.append(f"{feature_name}: {float(edge_attr[i, j]):.3f}")

                if feature_values:
                    edge["title"] = "\\n".join(feature_values)

            # Set edge width based on feature
            if edge_width_feature and hasattr(data, 'edge_attr') and data.edge_attr is not None:
                edge_attr = data.edge_attr.numpy() if isinstance(data.edge_attr, torch.Tensor) else data.edge_attr
                if edge_features and edge_width_feature in edge_features:
                    feature_idx = edge_features.index(edge_width_feature)
                    if feature_idx < edge_attr.shape[1]:
                        # Normalize width between 1 and 10
                        feature_val = float(edge_attr[i, feature_idx])
                        edge_min = float(np.min(edge_attr[:, feature_idx]))
                        edge_max = float(np.max(edge_attr[:, feature_idx]))
                        normalized_width = 1 + (feature_val - edge_min) / (edge_max - edge_min + 1e-8) * 9
                        edge["width"] = float(normalized_width)

            # Default edge styling
            edge["color"] = self.edge_colors[0]
            edge["arrows"] = "to"  # Directed edges

            edges.append(edge)

        return edges

    def _create_options(self, layout_method: str) -> Dict:
        """Create vis.js options"""
        options = {
            "physics": {
                "enabled": layout_method == 'physics',
                "solver": "barnesHut",
                "barnesHut": {
                    "gravitationalConstant": -8000,
                    "springConstant": 0.001,
                    "springLength": 200
                }
            },
            "layout": {
                "improvedLayout": True
            },
            "interaction": {
                "hover": True,
                "selectConnectedEdges": False
            },
            "nodes": {
                "shape": "dot",
                "size": 20,
                "font": {
                    "size": 14,
                    "color": "#000000"
                },
                "borderWidth": 2,
                "shadow": True
            },
            "edges": {
                "width": 2,
                "shadow": True,
                "smooth": {
                    "type": "continuous"
                }
            }
        }

        if layout_method == 'circular':
            options["layout"]["randomSeed"] = 2

        return options

    def save_to_json(self, visjs_data: Dict, filename: str):
        """Save vis.js data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(visjs_data, f, indent=2)
        print(f"Saved vis.js data to {filename}")


def create_sample_data():
    """Create a sample PyTorch Geometric Data object for testing"""
    # Create a simple graph: 5 nodes with some edges
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 4, 4, 0],
                               [1, 0, 2, 1, 3, 2, 4, 3, 0, 4]], dtype=torch.long)

    # Node features (5 nodes, 3 features each)
    x = torch.randn(5, 3)

    # Node labels (classification)
    y = torch.tensor([0, 1, 0, 1, 2], dtype=torch.long)

    # Edge features (10 edges, 2 features each)
    edge_attr = torch.randn(10, 2)

    data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)
    return data


class HTMLRenderer:
    """
    Renders PyTorch Geometric graphs as interactive HTML pages using vis.js
    """

    def __init__(self):
        self.converter = PyGToVisJS()

    def create_html_page(self,
                         data: Data,
                         title: str = "Graph Visualization",
                         node_features: Optional[List[str]] = None,
                         edge_features: Optional[List[str]] = None,
                         layout_method: str = 'physics',
                         node_size_feature: Optional[str] = None,
                         node_color_feature: Optional[str] = None,
                         edge_width_feature: Optional[str] = None,
                         width: str = "100%",
                         height: str = "600px") -> str:
        """
        Create a complete HTML page with vis.js graph visualization

        Args:
            data: PyTorch Geometric Data object
            title: Page title
            node_features: List of node feature names
            edge_features: List of edge feature names
            layout_method: Layout method for vis.js
            node_size_feature: Feature to use for node sizing
            node_color_feature: Feature to use for node coloring
            edge_width_feature: Feature to use for edge width
            width: Container width
            height: Container height

        Returns:
            Complete HTML string
        """

        # Convert PyG data to vis.js format
        visjs_data = self.converter.convert_data_to_visjs(
            data, node_features, edge_features, layout_method,
            node_size_feature, node_color_feature, edge_width_feature
        )

        # Create HTML template
        html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js"></script>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css" rel="stylesheet" type="text/css" />
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}

        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }}

        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 20px;
            text-align: center;
        }}

        .header h1 {{
            margin: 0;
            font-size: 24px;
        }}

        .stats {{
            background-color: #ecf0f1;
            padding: 15px 20px;
            border-bottom: 1px solid #bdc3c7;
        }}

        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
        }}

        .stat-item {{
            text-align: center;
        }}

        .stat-number {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}

        .stat-label {{
            font-size: 12px;
            color: #7f8c8d;
            text-transform: uppercase;
        }}

        #network-container {{
            width: {width};
            height: {height};
            border: 1px solid #bdc3c7;
        }}

        .controls {{
            padding: 20px;
            background-color: #ecf0f1;
            border-top: 1px solid #bdc3c7;
        }}

        .control-group {{
            display: inline-flex;
            align-items: center;
            margin-right: 20px;
            margin-bottom: 10px;
        }}

        .control-group label {{
            margin-right: 8px;
            font-weight: bold;
            color: #2c3e50;
        }}

        .control-group button {{
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 14px;
        }}

        .control-group button:hover {{
            background-color: #2980b9;
        }}

        .legend {{
            padding: 20px;
            background-color: #f8f9fa;
        }}

        .legend h3 {{
            margin-top: 0;
            color: #2c3e50;
        }}

        .legend-item {{
            display: inline-flex;
            align-items: center;
            margin-right: 20px;
            margin-bottom: 5px;
        }}

        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            margin-right: 8px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{title}</h1>
        </div>

        <div class="stats">
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-number">{len(visjs_data['nodes'])}</div>
                    <div class="stat-label">Nodes</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(visjs_data['edges'])}</div>
                    <div class="stat-label">Edges</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{len(visjs_data['edges']) / len(visjs_data['nodes']):.2f}</div>
                    <div class="stat-label">Avg Degree</div>
                </div>
                <div class="stat-item">
                    <div class="stat-number">{layout_method.title()}</div>
                    <div class="stat-label">Layout</div>
                </div>
            </div>
        </div>

        <div id="network-container"></div>

        <div class="controls">
            <div class="control-group">
                <label>Layout:</label>
                <button onclick="togglePhysics()">Toggle Physics</button>
            </div>
            <div class="control-group">
                <label>View:</label>
                <button onclick="fitNetwork()">Fit to Screen</button>
            </div>
            <div class="control-group">
                <label>Selection:</label>
                <button onclick="selectAll()">Select All</button>
                <button onclick="unselectAll()">Unselect All</button>
            </div>
        </div>

        {self._create_legend_html(visjs_data, node_features, edge_features)}
    </div>

    <script>
        // Graph data
        const nodes = new vis.DataSet({json.dumps(visjs_data['nodes'], indent=8)});
        const edges = new vis.DataSet({json.dumps(visjs_data['edges'], indent=8)});
        const data = {{ nodes: nodes, edges: edges }};

        // Options
        const options = {json.dumps(visjs_data['options'], indent=8)};

        // Create network
        const container = document.getElementById('network-container');
        const network = new vis.Network(container, data, options);

        // Event listeners
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                const nodeId = params.nodes[0];
                const node = nodes.get(nodeId);
                console.log('Clicked node:', node);
            }}
            if (params.edges.length > 0) {{
                const edgeId = params.edges[0];
                const edge = edges.get(edgeId);
                console.log('Clicked edge:', edge);
            }}
        }});

        network.on('hoverNode', function(params) {{
            container.style.cursor = 'pointer';
        }});

        network.on('blurNode', function(params) {{
            container.style.cursor = 'default';
        }});

        // Control functions
        function togglePhysics() {{
            const currentPhysics = network.physics.physicsEnabled;
            network.setOptions({{ physics: {{ enabled: !currentPhysics }} }});
        }}

        function fitNetwork() {{
            network.fit();
        }}

        function selectAll() {{
            const allNodes = nodes.getIds();
            network.selectNodes(allNodes);
        }}

        function unselectAll() {{
            network.unselectAll();
        }}

        // Initial fit
        network.once('stabilized', function() {{
            network.fit();
        }});
    </script>
</body>
</html>"""

        return html_template

    def _create_legend_html(self, visjs_data: Dict, node_features: Optional[List[str]],
                            edge_features: Optional[List[str]]) -> str:
        """Create legend HTML section"""
        if not node_features and not edge_features:
            return ""

        legend_html = '<div class="legend"><h3>Legend</h3>'

        if node_features:
            legend_html += '<p><strong>Node Features:</strong> ' + ', '.join(node_features) + '</p>'

        if edge_features:
            legend_html += '<p><strong>Edge Features:</strong> ' + ', '.join(edge_features) + '</p>'

        # Add color legend for node classes
        unique_colors = list(set(node.get('color', '#97C2FC') for node in visjs_data['nodes']))
        if len(unique_colors) > 1:
            legend_html += '<p><strong>Node Colors:</strong></p>'
            for i, color in enumerate(unique_colors):
                legend_html += f'<div class="legend-item"><div class="legend-color" style="background-color: {color};"></div>Class {i}</div>'

        legend_html += '</div>'
        return legend_html

    def render_to_file(self,
                       data: Data,
                       filename: str,
                       title: str = "Graph Visualization",
                       **kwargs):
        """
        Render graph to HTML file

        Args:
            data: PyTorch Geometric Data object
            filename: Output HTML filename
            title: Page title
            **kwargs: Additional arguments for create_html_page
        """
        html_content = self.create_html_page(data, title=title, **kwargs)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_content)

        print(f"HTML visualization saved to {filename}")
        return filename


# Example usage
if __name__ == "__main__":
    # Create renderer
    renderer = HTMLRenderer()

    # Create sample data
    #sample_data = create_sample_data()

    filepath = f"{Path(__file__).parents[3]}/game_strings/combined.txt"
    batch_size = 10
    loader = GameDataLoader(filepath, batch_size=batch_size)
    total_batches = (len(loader) + batch_size - 1) // batch_size

    # load first game
    game = loader.get_game(10)

    # process the endgame
    all_data = process_endgame(game, include_moves=True, include_value=True)


    # Render to HTML file
    renderer.render_to_file(
        all_data[0],
        'graph_visualization.html',
        title="PyTorch Geometric Graph Visualization",
        node_features=[''],
        edge_features=['edge_weight'],
        layout_method='physics',
        node_size_feature=None,
        node_color_feature='class',
        edge_width_feature=None,
        height='700px'
    )

    print("Open 'graph_visualization.html' in your browser to view the interactive visualization!")