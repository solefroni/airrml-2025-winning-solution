#!/usr/bin/env python3
"""
Graph Builder using KNN in CVC Embedding Space
===============================================

Builds graphs from TCR repertoires using KNN in CVC embedding space.
Uses FAISS for efficient nearest neighbor search.
"""

import numpy as np
import torch
from torch_geometric.data import Data
import networkx as nx
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Import FAISS at module level, but check GPU at runtime
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    faiss = None
    FAISS_AVAILABLE = False
    FAISS_GPU_AVAILABLE = False
    print("Warning: FAISS not available. Using PyTorch for KNN (slower)")

# Import HNSWlib for fast approximate nearest neighbor search
try:
    import hnswlib
    HNSWLIB_AVAILABLE = True
except ImportError:
    hnswlib = None
    HNSWLIB_AVAILABLE = False

FAISS_GPU_AVAILABLE = None

def _check_faiss_gpu():
    """Check FAISS GPU availability at runtime."""
    global FAISS_GPU_AVAILABLE
    if FAISS_GPU_AVAILABLE is not None:
        return FAISS_GPU_AVAILABLE
    
    if not FAISS_AVAILABLE or faiss is None:
        FAISS_GPU_AVAILABLE = False
        return False
    
    FAISS_GPU_AVAILABLE = False
    if hasattr(faiss, 'StandardGpuResources'):
        try:
            # Try to create GPU resources to verify GPU support
            gpu_res = faiss.StandardGpuResources()
            # Test with a small index
            test_index = faiss.IndexFlatL2(10)
            gpu_index = faiss.index_cpu_to_gpu(gpu_res, 0, test_index)
            FAISS_GPU_AVAILABLE = True
        except Exception:
            FAISS_GPU_AVAILABLE = False
    return FAISS_GPU_AVAILABLE

from cvc_embedder import CVCEmbedder


class GraphBuilder:
    """Builds graphs from TCR repertoires using KNN in embedding space."""
    
    def __init__(
        self,
        embedder,  # Can be CVCEmbedder or TCRformerEmbedder
        k_neighbors: int = 20,
        distance_metric: str = "cosine",
        device: str = "cuda",
        verbose: bool = True,
        use_faiss: bool = True,
        use_hnswlib: bool = True
    ):
        """
        Initialize graph builder.
        
        Args:
            embedder: Embedder instance (CVCEmbedder or TCRformerEmbedder)
            k_neighbors: Number of nearest neighbors for KNN
            distance_metric: "cosine" or "euclidean"
            device: torch device
            verbose: Print status messages
            use_faiss: Whether to use FAISS for KNN (False = use PyTorch KNN)
            use_hnswlib: Whether to use HNSWlib for KNN (fallback if FAISS unavailable)
        """
        self.embedder = embedder
        # Keep backward compatibility
        self.cvc_embedder = embedder
        self.k_neighbors = k_neighbors
        self.distance_metric = distance_metric
        # Normalize device string: 'cuda:1' -> 'cuda:1', 'cuda' -> 'cuda'
        self.device = device if isinstance(device, str) else str(device)
        self.verbose = verbose
        self.use_faiss = use_faiss
        self.use_hnswlib = use_hnswlib
        
        # Reuse FAISS GPU resources to avoid memory fragmentation
        self._faiss_gpu_res = None
        if self.use_faiss and ('cuda' in self.device) and FAISS_AVAILABLE and faiss is not None:
            if _check_faiss_gpu():
                try:
                    self._faiss_gpu_res = faiss.StandardGpuResources()
                    # Set temp memory to a reasonable limit (512MB instead of default 2GB)
                    # This helps prevent OOM when GPU memory is fragmented
                    self._faiss_gpu_res.setTempMemory(512 * 1024 * 1024)  # 512MB
                except Exception as e:
                    if self.verbose:
                        print(f"  Warning: Could not initialize FAISS GPU resources: {e}")
                    self._faiss_gpu_res = None
        
        if self.verbose:
            if self._faiss_gpu_res is not None:
                knn_backend = "FAISS GPU"
            elif self.use_hnswlib and HNSWLIB_AVAILABLE:
                knn_backend = "HNSWlib (CPU, fast approximate)"
            elif not self.use_faiss:
                if 'cuda' in self.device:
                    knn_backend = "PyTorch GPU (FAISS/HNSWlib disabled)"
                else:
                    knn_backend = "PyTorch CPU (FAISS/HNSWlib disabled)"
            elif 'cuda' in self.device:
                knn_backend = "PyTorch GPU"
            else:
                knn_backend = "PyTorch CPU"
            print(f"GraphBuilder: k={k_neighbors}, metric={distance_metric}, KNN={knn_backend}")
    
    def build_graph(
        self,
        sequences: List[str],
        counts: np.ndarray,
        node_features: Optional[Dict] = None,
        use_cache: bool = True,
        v_genes: Optional[List[str]] = None,
        j_genes: Optional[List[str]] = None
    ) -> Data:
        """
        Build a graph from sequences using KNN in embedding space.
        
        Args:
            sequences: List of CDR3 sequences
            counts: Array of template counts for each sequence
            node_features: Optional dict of additional node features
            use_cache: Whether to use cached embeddings
            v_genes: Optional list of V gene names (for TCRformer)
            j_genes: Optional list of J gene names (for TCRformer)
        
        Returns:
            PyTorch Geometric Data object
        """
        if len(sequences) == 0:
            raise ValueError("Cannot build graph from empty sequence list")
        
        # Get embeddings (with caching)
        # Try to load from cache first
        embeddings = None
        if use_cache:
            from cache_utils import load_cached_embeddings, save_cached_embeddings
            # For TCRformer, cache key should include V/J genes if available
            cache_key = sequences
            if v_genes is not None and j_genes is not None:
                # Include V/J genes in cache key for TCRformer
                cache_key = (sequences, tuple(v_genes), tuple(j_genes))
            
            embeddings = load_cached_embeddings(cache_key, None)
        
        if embeddings is None:
            if self.verbose:
                print(f"  Generating embeddings for {len(sequences)} sequences...")
            # Check if embedder supports V/J genes (TCRformer)
            if hasattr(self.embedder, 'embed') and v_genes is not None and j_genes is not None:
                try:
                    embeddings = self.embedder.embed(sequences, v_genes=v_genes, j_genes=j_genes)
                except TypeError:
                    # Fallback if embedder doesn't support V/J genes
                    embeddings = self.embedder.embed(sequences)
            else:
                embeddings = self.embedder.embed(sequences)
            embeddings = embeddings.astype(np.float32)
            # Save to cache
            if use_cache:
                from cache_utils import save_cached_embeddings
                cache_key = sequences
                if v_genes is not None and j_genes is not None:
                    cache_key = (sequences, tuple(v_genes), tuple(j_genes))
                
                save_cached_embeddings(cache_key, embeddings, None)
        else:
            if self.verbose:
                print(f"  Using cached embeddings for {len(sequences)} sequences")
        
        embeddings = embeddings.astype(np.float32)
        
        # Build KNN graph
        if self.verbose:
            print(f"  Building KNN graph (k={self.k_neighbors})...")
        
        edge_index, edge_attr = self._build_knn_graph(embeddings)
        
        # Prepare node features
        node_feat = self._prepare_node_features(
            sequences, counts, embeddings, node_features
        )
        
        # Create PyTorch Geometric Data object (always on CPU for DataLoader compatibility)
        # DataLoader will move to device during training
        data = Data(
            x=torch.tensor(node_feat, dtype=torch.float32, device='cpu'),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(sequences)
        )
        
        return data
    
    def build_graph_from_embeddings(
        self,
        sequences: List[str],
        counts: np.ndarray,
        embeddings: np.ndarray,
        node_features: Optional[Dict] = None
    ) -> Data:
        """
        Build a graph from pre-computed embeddings (for batch processing optimization).
        
        Args:
            sequences: List of CDR3 sequences
            counts: Array of template counts for each sequence
            embeddings: Pre-computed embeddings array (n_sequences, embedding_dim)
            node_features: Optional dict of additional node features
        
        Returns:
            PyTorch Geometric Data object
        """
        if len(sequences) == 0:
            raise ValueError("Cannot build graph from empty sequence list")
        
        if len(sequences) != len(embeddings):
            raise ValueError(f"Sequence count ({len(sequences)}) doesn't match embedding count ({len(embeddings)})")
        
        embeddings = embeddings.astype(np.float32)
        
        # Build KNN graph
        if self.verbose:
            print(f"  Building KNN graph (k={self.k_neighbors})...")
        
        edge_index, edge_attr = self._build_knn_graph(embeddings)
        
        # Prepare node features
        node_feat = self._prepare_node_features(
            sequences, counts, embeddings, node_features
        )
        
        # Create PyTorch Geometric Data object (always on CPU for DataLoader compatibility)
        data = Data(
            x=torch.tensor(node_feat, dtype=torch.float32, device='cpu'),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(sequences)
        )
        
        return data
    
    def _build_knn_graph(
        self,
        embeddings: np.ndarray
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Build KNN graph from embeddings.
        
        Returns:
            edge_index: (2, num_edges) tensor
            edge_attr: (num_edges,) tensor of edge weights (distances)
        """
        n = len(embeddings)
        k = min(self.k_neighbors + 1, n)  # +1 because point is its own neighbor
        
        # Special handling for k=1 (each node connects to exactly 1 neighbor)
        if self.k_neighbors == 1:
            k = 2  # Get 2 neighbors (self + 1 other), then remove self
        
        # Clear GPU cache before KNN to free up memory
        if 'cuda' in self.device:
            # Get device index if specified (e.g., 'cuda:1' -> 1)
            device_idx = 0
            if ':' in self.device:
                try:
                    device_idx = int(self.device.split(':')[1])
                except:
                    device_idx = 0
            torch.cuda.set_device(device_idx)
            torch.cuda.empty_cache()
        
        # Try HNSWlib first (fast, reliable, CPU-based approximate search)
        hnswlib_success = False
        hnswlib_distances = None
        hnswlib_indices = None
        if self.use_hnswlib and HNSWLIB_AVAILABLE and hnswlib is not None:
            try:
                # HNSWlib works on CPU but is still fast
                n = len(embeddings)
                dim = embeddings.shape[1]
                k_search = min(self.k_neighbors + 1, n)  # +1 because point is its own neighbor
                
                # Create index
                space = 'cosine' if self.distance_metric == "cosine" else 'l2'
                index = hnswlib.Index(space=space, dim=dim)
                
                # Initialize with parameters optimized for speed vs accuracy
                # ef_construction: controls index construction time/quality (higher = better but slower)
                # M: number of bi-directional links (higher = better recall but more memory)
                ef_construction = min(200, max(100, n // 50))  # Adaptive based on dataset size
                M = 16  # Good balance for 10k sequences
                index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
                
                # Add embeddings
                index.add_items(embeddings.astype(np.float32), np.arange(n))
                
                # Set ef for search (should be >= k, higher = more accurate but slower)
                ef_search = max(k_search * 2, 100)  # At least 2x k for good accuracy
                index.set_ef(ef_search)
                
                # Search for all points (each point searches for its neighbors)
                labels, distances = index.knn_query(embeddings.astype(np.float32), k=k_search)
                
                hnswlib_indices = labels.astype(np.int64)
                
                # Convert distances: HNSWlib returns similarity for cosine, distance for l2
                if self.distance_metric == "cosine":
                    # For cosine, HNSWlib returns 1 - cosine_similarity, so convert to distance
                    hnswlib_distances = distances.astype(np.float32)
                else:
                    # For l2, distances are already correct
                    hnswlib_distances = distances.astype(np.float32)
                
                hnswlib_success = True
                
                # Clean up
                del index
                
            except Exception as e:
                if self.verbose:
                    print(f"  HNSWlib error ({type(e).__name__}), trying other methods: {e}")
                hnswlib_success = False
        
        # Try FAISS GPU second (if available and resources initialized and device is CUDA)
        faiss_success = False
        faiss_distances = None
        faiss_indices = None
        if not hnswlib_success and self.use_faiss and FAISS_AVAILABLE and faiss is not None and self._faiss_gpu_res is not None and ('cuda' in self.device):
            try:
                # Ensure embeddings are float32 and contiguous for FAISS
                embeddings_faiss = np.ascontiguousarray(embeddings.astype(np.float32))
                
                if self.distance_metric == "cosine":
                    # Normalize for cosine distance (in-place)
                    faiss.normalize_L2(embeddings_faiss)
                    index_cpu = faiss.IndexFlatIP(embeddings_faiss.shape[1])  # Inner product for cosine
                else:
                    index_cpu = faiss.IndexFlatL2(embeddings_faiss.shape[1])
                
                # Move to GPU using reused resources
                # Get device index if specified (e.g., 'cuda:1' -> 1)
                device_idx = 0
                if ':' in self.device:
                    try:
                        device_idx = int(self.device.split(':')[1])
                    except:
                        device_idx = 0
                
                index = faiss.index_cpu_to_gpu(self._faiss_gpu_res, device_idx, index_cpu)
                index.add(embeddings_faiss)
                faiss_distances, faiss_indices = index.search(embeddings_faiss, k)
                
                # Clean up FAISS index immediately to free GPU memory
                del index
                del index_cpu
                del embeddings_faiss  # Free embeddings copy
                torch.cuda.empty_cache()
                import gc
                gc.collect()  # Force garbage collection
                
                faiss_success = True
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower() or "cudaMalloc" in str(e):
                    # GPU OOM - use PyTorch GPU instead (more memory efficient)
                    if self.verbose:
                        print(f"  FAISS GPU OOM, using PyTorch GPU KNN instead")
                    torch.cuda.empty_cache()
                else:
                    raise
            except Exception as e:
                # Other FAISS errors - use PyTorch GPU instead
                if self.verbose:
                    print(f"  FAISS GPU error ({type(e).__name__}), using PyTorch GPU KNN instead")
                torch.cuda.empty_cache()
        
        # Process HNSWlib results if successful
        if hnswlib_success:
            # Convert HNSWlib results to numpy arrays
            distances = hnswlib_distances.astype(np.float32)
            indices = hnswlib_indices.astype(np.int64)
            
            # Remove self-connections
            mask = indices != np.arange(n)[:, None]
            num_neighbors = k - 1
            if num_neighbors > 0:
                indices_clean = np.zeros((n, num_neighbors), dtype=indices.dtype)
                distances_clean = np.zeros((n, num_neighbors), dtype=distances.dtype)
                
                for i in range(n):
                    # Get neighbors for this node, excluding self
                    node_neighbors = indices[i][indices[i] != i]
                    node_distances = distances[i][indices[i] != i]
                    
                    # Take exactly num_neighbors neighbors
                    if len(node_neighbors) >= num_neighbors:
                        indices_clean[i] = node_neighbors[:num_neighbors]
                        distances_clean[i] = node_distances[:num_neighbors]
                    else:
                        # Pad with last neighbor if we have fewer than num_neighbors
                        if len(node_neighbors) > 0:
                            indices_clean[i, :len(node_neighbors)] = node_neighbors
                            indices_clean[i, len(node_neighbors):] = node_neighbors[-1]
                            distances_clean[i, :len(node_distances)] = node_distances
                            distances_clean[i, len(node_distances):] = node_distances[-1]
                        else:
                            # No neighbors (shouldn't happen), use self as fallback
                            indices_clean[i] = i
                            distances_clean[i] = 0.0
            else:
                # Fallback for k=1
                indices_clean = indices[:, 1:2]
                distances_clean = distances[:, 1:2]
                num_neighbors = 1
            
            # Build edges
            src_nodes = np.repeat(np.arange(n), num_neighbors)
            dst_nodes = indices_clean.flatten()
            valid_mask = dst_nodes < n
            src_nodes = src_nodes[valid_mask]
            dst_nodes = dst_nodes[valid_mask]
            
            # Create bidirectional edges
            edges_forward = np.stack([src_nodes, dst_nodes], axis=0)
            edges_backward = np.stack([dst_nodes, src_nodes], axis=0)
            edges = np.concatenate([edges_forward, edges_backward], axis=1)
            
            # Compute edge weights (inverse distance)
            valid_distances = distances_clean.flatten()[valid_mask]
            weights = 1.0 / (1.0 + valid_distances)
            edge_weights = np.concatenate([weights, weights])  # Duplicate for bidirectional
            
            # Convert to tensors
            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
            
            # Clean up numpy arrays
            del edges, edges_forward, edges_backward, edge_weights, distances, indices, hnswlib_distances, hnswlib_indices
            
            return edge_index.contiguous(), edge_attr
        
        # Process FAISS results if successful
        if faiss_success:
            # Convert FAISS results to numpy arrays
            distances = faiss_distances.astype(np.float32)
            indices = faiss_indices.astype(np.int64)
            
            # For cosine distance, convert inner product to distance
            if self.distance_metric == "cosine":
                distances = 1.0 - distances  # Convert similarity to distance
            
            # Remove self-connections
            mask = indices != np.arange(n)[:, None]
            num_neighbors = k - 1
            if num_neighbors > 0:
                indices_clean = np.zeros((n, num_neighbors), dtype=indices.dtype)
                distances_clean = np.zeros((n, num_neighbors), dtype=distances.dtype)
                
                for i in range(n):
                    # Get neighbors for this node, excluding self
                    node_neighbors = indices[i][indices[i] != i]
                    node_distances = distances[i][indices[i] != i]
                    
                    # Take exactly num_neighbors neighbors
                    if len(node_neighbors) >= num_neighbors:
                        indices_clean[i] = node_neighbors[:num_neighbors]
                        distances_clean[i] = node_distances[:num_neighbors]
                    else:
                        # Pad with last neighbor if we have fewer than num_neighbors
                        if len(node_neighbors) > 0:
                            indices_clean[i, :len(node_neighbors)] = node_neighbors
                            indices_clean[i, len(node_neighbors):] = node_neighbors[-1]
                            distances_clean[i, :len(node_distances)] = node_distances
                            distances_clean[i, len(node_distances):] = node_distances[-1]
                        else:
                            # No neighbors (shouldn't happen), use self as fallback
                            indices_clean[i] = i
                            distances_clean[i] = 0.0
            else:
                # Fallback for k=1
                indices_clean = indices[:, 1:2]
                distances_clean = distances[:, 1:2]
                num_neighbors = 1
            
            # Build edges
            src_nodes = np.repeat(np.arange(n), num_neighbors)
            dst_nodes = indices_clean.flatten()
            valid_mask = dst_nodes < n
            src_nodes = src_nodes[valid_mask]
            dst_nodes = dst_nodes[valid_mask]
            
            # Create bidirectional edges
            edges_forward = np.stack([src_nodes, dst_nodes], axis=0)
            edges_backward = np.stack([dst_nodes, src_nodes], axis=0)
            edges = np.concatenate([edges_forward, edges_backward], axis=1)
            
            # Compute edge weights (inverse distance)
            valid_distances = distances_clean.flatten()[valid_mask]
            weights = 1.0 / (1.0 + valid_distances)
            edge_weights = np.concatenate([weights, weights])  # Duplicate for bidirectional
            
            # Convert to tensors
            edge_index = torch.tensor(edges, dtype=torch.long)
            edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
            
            # Clean up numpy arrays
            del edges, edges_forward, edges_backward, edge_weights, distances, indices, faiss_distances, faiss_indices
            torch.cuda.empty_cache()
            
            return edge_index.contiguous(), edge_attr
        
        # Use PyTorch GPU KNN (either as primary or fallback from HNSWlib/FAISS)
        if not hnswlib_success and not faiss_success and ('cuda' in self.device):
            # PyTorch GPU path (more memory efficient, no CPU fallback)
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32, device=self.device)
            
            if self.distance_metric == "cosine":
                # Normalize
                embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
                # Compute cosine similarity
                similarity = torch.mm(embeddings_tensor, embeddings_tensor.t())
                distances_tensor, indices_tensor = torch.topk(similarity, k, dim=1)
                distances_tensor = 1 - distances_tensor  # Convert similarity to distance
            else:
                # Euclidean distance
                distances_matrix = torch.cdist(embeddings_tensor, embeddings_tensor)
                distances_tensor, indices_tensor = torch.topk(distances_matrix, k, dim=1, largest=False)
            
            # Remove self-connections (keep on GPU)
            mask = indices_tensor != torch.arange(n, device=self.device)[:, None]
            # For k=1, we need exactly 1 neighbor per node (k-1=1 after removing self)
            num_neighbors = k - 1
            if num_neighbors > 0:
                indices_gpu = indices_tensor[mask].reshape(n, num_neighbors)
                distances_gpu = distances_tensor[mask].reshape(n, num_neighbors)
            else:
                # Fallback: if somehow we have 0 neighbors, use the first non-self neighbor
                # This shouldn't happen with our k adjustment, but keep for safety
                indices_gpu = indices_tensor[:, 1:2]  # Take second column (first non-self)
                distances_gpu = distances_tensor[:, 1:2]
                num_neighbors = 1
            
            # Build edges directly on GPU (no CPU transfer)
            # Create source nodes: [0,0,0,...,1,1,1,...,n-1,n-1,n-1]
            src_nodes = torch.repeat_interleave(torch.arange(n, device=self.device), num_neighbors)
            # Create target nodes from indices (flatten)
            dst_nodes = indices_gpu.flatten()
            # Filter out invalid indices
            valid_mask = dst_nodes < n
            src_nodes = src_nodes[valid_mask]
            dst_nodes = dst_nodes[valid_mask]
            
            # Create bidirectional edges on GPU: [src->dst, dst->src]
            edges_forward = torch.stack([src_nodes, dst_nodes], dim=0)
            edges_backward = torch.stack([dst_nodes, src_nodes], dim=0)
            edge_index = torch.cat([edges_forward, edges_backward], dim=1)
            
            # Compute edge weights on GPU (inverse distance)
            valid_distances = distances_gpu.flatten()[valid_mask]
            weights = 1.0 / (1.0 + valid_distances)
            # Duplicate for bidirectional
            edge_attr = torch.cat([weights, weights])
            
            # Move to CPU for DataLoader compatibility (will be moved to device during training)
            edge_index = edge_index.cpu().contiguous()
            edge_attr = edge_attr.cpu()
            
            # Clean up intermediate tensors
            del embeddings_tensor, distances_tensor, indices_tensor, distances_gpu, indices_gpu
            torch.cuda.empty_cache()
            
            return edge_index, edge_attr
        
        # PyTorch CPU path (for k=1 or when CUDA not available)
        if self.device == 'cpu' or (not ('cuda' in self.device)):
            embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
            
            if self.distance_metric == "cosine":
                # Normalize
                embeddings_tensor = torch.nn.functional.normalize(embeddings_tensor, p=2, dim=1)
                # Compute cosine similarity
                similarity = torch.mm(embeddings_tensor, embeddings_tensor.t())
                distances_tensor, indices_tensor = torch.topk(similarity, k, dim=1)
                distances_tensor = 1 - distances_tensor  # Convert similarity to distance
            else:
                # Euclidean distance
                distances_matrix = torch.cdist(embeddings_tensor, embeddings_tensor)
                distances_tensor, indices_tensor = torch.topk(distances_matrix, k, dim=1, largest=False)
            
            # Remove self-connections
            mask = indices_tensor != torch.arange(n)[:, None]
            num_neighbors = k - 1
            if num_neighbors > 0:
                indices_cpu = indices_tensor[mask].reshape(n, num_neighbors)
                distances_cpu = distances_tensor[mask].reshape(n, num_neighbors)
            else:
                # Fallback for k=1
                indices_cpu = indices_tensor[:, 1:2]
                distances_cpu = distances_tensor[:, 1:2]
                num_neighbors = 1
            
            # Build edges
            src_nodes = torch.repeat_interleave(torch.arange(n), num_neighbors)
            dst_nodes = indices_cpu.flatten()
            valid_mask = dst_nodes < n
            src_nodes = src_nodes[valid_mask]
            dst_nodes = dst_nodes[valid_mask]
            
            # Create bidirectional edges
            edges_forward = torch.stack([src_nodes, dst_nodes], dim=0)
            edges_backward = torch.stack([dst_nodes, src_nodes], dim=0)
            edge_index = torch.cat([edges_forward, edges_backward], dim=1)
            
            # Compute edge weights
            valid_distances = distances_cpu.flatten()[valid_mask]
            weights = 1.0 / (1.0 + valid_distances)
            edge_attr = torch.cat([weights, weights])
            
            return edge_index.contiguous(), edge_attr
        
        # FAISS path - build edges efficiently then convert to GPU tensors
        # (Only reached if FAISS succeeded)
        # Note: If FAISS failed and device is CUDA, PyTorch GPU path should have been used above
        # If device is CPU, CPU path should have been used above
        # So this code should only execute if FAISS succeeded
        if not faiss_success:
            # This should not happen - either CPU or PyTorch GPU path should have been used
            raise RuntimeError(f"Unexpected: FAISS failed but CPU/GPU paths not used (device={self.device})")
        
        # Remove self-connections first - do it row by row to ensure exactly k-1 per node
        num_neighbors = k - 1
        if num_neighbors <= 0:
            # Fallback for k=1 case (shouldn't happen with our k adjustment)
            num_neighbors = 1
        
        indices_clean = np.zeros((n, num_neighbors), dtype=indices.dtype)
        distances_clean = np.zeros((n, num_neighbors), dtype=distances.dtype)
        
        for i in range(n):
            # Get neighbors for this node, excluding self
            node_neighbors = indices[i][indices[i] != i]
            node_distances = distances[i][indices[i] != i]
            
            # Take exactly num_neighbors neighbors (in case of duplicates)
            if len(node_neighbors) >= num_neighbors:
                indices_clean[i] = node_neighbors[:num_neighbors]
                distances_clean[i] = node_distances[:num_neighbors]
            else:
                # Pad with last neighbor if we have fewer than num_neighbors
                if len(node_neighbors) > 0:
                    indices_clean[i, :len(node_neighbors)] = node_neighbors
                    indices_clean[i, len(node_neighbors):] = node_neighbors[-1]
                    distances_clean[i, :len(node_distances)] = node_distances
                    distances_clean[i, len(node_distances):] = node_distances[-1]
                else:
                    # No neighbors (shouldn't happen), use self as fallback
                    indices_clean[i] = i
                    distances_clean[i] = 0.0
        
        indices = indices_clean
        distances = distances_clean
        
        # Build edge list (bidirectional) - optimized vectorized version
        # Create source nodes: [0,0,0,...,1,1,1,...,n-1,n-1,n-1]
        src_nodes = np.repeat(np.arange(n), num_neighbors)
        # Create target nodes from indices (flatten)
        dst_nodes = indices.flatten()
        # Filter out invalid indices (shouldn't be needed, but keep for safety)
        valid_mask = dst_nodes < n
        src_nodes = src_nodes[valid_mask]
        dst_nodes = dst_nodes[valid_mask]
        
        # Create bidirectional edges: [src->dst, dst->src]
        # Use vstack for better memory layout
        edges_forward = np.vstack([src_nodes, dst_nodes])
        edges_backward = np.vstack([dst_nodes, src_nodes])
        edges = np.hstack([edges_forward, edges_backward])
        
        # Compute edge weights (inverse distance)
        valid_distances = distances.flatten()[valid_mask]
        weights = 1.0 / (1.0 + valid_distances)
        # Duplicate for bidirectional
        edge_weights = np.concatenate([weights, weights])
        
        # Convert to GPU tensors directly (single transfer)
        # Create on CPU for DataLoader compatibility (will be moved to device during training)
        edge_index = torch.from_numpy(edges).long().contiguous()
        edge_attr = torch.from_numpy(edge_weights).float()
        
        # Clean up numpy arrays
        del edges, edges_forward, edges_backward, edge_weights, distances, indices
        torch.cuda.empty_cache()
        
        return edge_index, edge_attr
    
    def _prepare_node_features(
        self,
        sequences: List[str],
        counts: np.ndarray,
        embeddings: np.ndarray,
        additional_features: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Prepare node features: frequency, centrality, sequence features, etc.
        
        Returns:
            numpy array of shape (n_nodes, n_features)
        """
        n = len(sequences)
        features = []
        
        # 1. Frequency (normalized count)
        total_count = counts.sum()
        frequencies = counts / total_count if total_count > 0 else np.zeros(n)
        features.append(frequencies.reshape(-1, 1))
        
        # 2. Sequence length (vectorized)
        seq_lengths = np.array([len(seq) for seq in sequences], dtype=np.float32)
        features.append(seq_lengths.reshape(-1, 1))
        
        # 3. Amino acid composition (20 standard AAs)
        aa_composition = self._compute_aa_composition(sequences)
        features.append(aa_composition)
        
        # 4. Centrality metrics (will be computed after graph is built)
        # For now, add placeholders
        centrality_placeholder = np.zeros((n, 4))  # degree, eigenvector (removed), pagerank (removed), clustering
        features.append(centrality_placeholder)
        
        # 5. Local graph density (placeholder, computed after graph)
        density_placeholder = np.zeros((n, 1))
        features.append(density_placeholder)
        
        # 6. CVC embedding (first 32 dimensions to keep feature size manageable)
        embedding_features = embeddings[:, :32]
        features.append(embedding_features)
        
        # Combine all features
        node_feat = np.hstack(features)
        
        return node_feat
    
    def _compute_aa_composition(self, sequences: List[str]) -> np.ndarray:
        """Compute amino acid composition for each sequence - optimized vectorized version."""
        # Standard 20 amino acids
        aa_list = 'ACDEFGHIKLMNPQRSTVWY'
        n = len(sequences)
        composition = np.zeros((n, len(aa_list)))
        
        # Create mapping dictionary for faster lookup
        aa_to_idx = {aa: idx for idx, aa in enumerate(aa_list)}
        
        # Vectorized computation
        for i, seq in enumerate(sequences):
            if len(seq) > 0:
                # Count amino acids using vectorized operations
                seq_array = np.array(list(seq))
                for aa, idx in aa_to_idx.items():
                    composition[i, idx] = np.sum(seq_array == aa)
                # Normalize by sequence length
                composition[i] /= len(seq)
        
        return composition
    
    def add_centrality_features(
        self,
        data: Data,
        compute_density: bool = True
    ) -> Data:
        """
        Add centrality features to graph nodes using GPU-accelerated PyTorch operations.
        Computes: degree, eigenvector (removed), PageRank (removed), clustering (GPU).
        
        Args:
            data: PyTorch Geometric Data object
            compute_density: Also compute local graph density
        
        Returns:
            Updated Data object with centrality features
        """
        n = data.num_nodes
        
        # Use GPU if available
        device = data.edge_index.device
        
        # Build adjacency matrix on GPU (optimized: use scatter_add for large graphs)
        edge_index = data.edge_index
        # For k=1, graphs are very sparse (1 neighbor per node), so skip dense matrix
        # For larger k or smaller graphs, dense matrix can be faster
        if n < 100000 and self.k_neighbors > 1:  # Only use dense for k>1 and smaller graphs
            adj = torch.sparse_coo_tensor(
                edge_index,
                torch.ones(edge_index.shape[1], device=device),
                size=(n, n)
            ).to_dense()
        else:
            # For k=1 or very large graphs, use sparse operations directly
            # Skip dense conversion for memory efficiency and speed
            adj = None
        
        # 1. Degree centrality (GPU-accelerated)
        if adj is not None:
            degree = adj.sum(dim=1).float()
        else:
            # For large graphs, compute degree from edge_index directly (faster)
            degree = torch.zeros(n, device=device, dtype=torch.float32)
            degree.scatter_add_(0, edge_index[0], torch.ones(edge_index.shape[1], device=device))
            # Divide by 2 since edges are bidirectional
            degree = degree / 2.0
        degree_centrality = degree / (n - 1) if n > 1 else degree
        
        # 2. Eigenvector centrality - REMOVED (too slow)
        eigenvector_centrality = torch.zeros(n, device=device, dtype=torch.float32)
        
        # 3. PageRank - REMOVED (too slow)
        pagerank = torch.zeros(n, device=device, dtype=torch.float32)
        
        # 4. Clustering coefficient (GPU-accelerated using matrix operations)
        # Clustering = (number of triangles) / (number of possible triangles)
        # For node i: triangles = sum of (A^3)[i,i] / 2, possible = degree*(degree-1)/2
        # OPTIMIZED: Skip clustering for larger/denser graphs (k>=10) to speed up graph building
        # Clustering is expensive: O(n^2 * k) for dense graphs
        if adj is not None and n < 50000 and self.k_neighbors < 10:  # Only for smaller k and graphs that fit in GPU memory
            adj_sq = torch.mm(adj, adj)
            adj_cube = torch.mm(adj_sq, adj)
            triangles = torch.diag(adj_cube) / 2.0  # Each triangle counted twice
            possible = degree * (degree - 1) / 2.0
            clustering = torch.where(possible > 0, triangles / possible, torch.zeros_like(degree))
        else:
            # For larger k values or very large graphs, skip clustering (too expensive)
            # This significantly speeds up graph building for k>=10
            clustering = torch.zeros(n, device=device, dtype=torch.float32)
        
        # 5. Local graph density (average degree of neighbors) - GPU-accelerated
        density = torch.zeros(n, device=device, dtype=torch.float32)
        if compute_density:
            if adj is not None:
                # Average degree of neighbors = (A @ degree) / degree
                neighbor_degree_sum = torch.mm(adj, degree.unsqueeze(1)).squeeze()
            else:
                # For large graphs, compute from edge_index using scatter_add
                neighbor_degree_sum = torch.zeros(n, device=device, dtype=torch.float32)
                # Get neighbor degrees: for each edge, add degree of neighbor
                neighbor_degrees = degree[edge_index[1]]
                neighbor_degree_sum.scatter_add_(0, edge_index[0], neighbor_degrees)
                neighbor_degree_sum = neighbor_degree_sum / 2.0  # Divide by 2 for bidirectional
            density = torch.where(degree > 0, neighbor_degree_sum / degree, torch.zeros_like(degree))
            density = density / (n - 1) if n > 1 else density
        
        # Update node features directly on GPU (no CPU transfer needed)
        # Stack centrality features on GPU
        centrality_features_gpu = torch.stack([
            degree_centrality,
            eigenvector_centrality,
            pagerank,
            clustering
        ], dim=1)  # Keep on GPU
        
        # Update node features in-place on GPU
        # Features are: [freq(1), length(1), aa_comp(20), centrality(4), density(1), embedding(32)]
        # The indices -37:-33 cover the 4 centrality features, -33 is density
        data.x[:, -37:-33] = centrality_features_gpu
        data.x[:, -33] = density
        
        return data
    
    def _to_networkx(self, data: Data) -> nx.Graph:
        """Convert PyTorch Geometric Data to NetworkX graph (deprecated - using GPU now)."""
        # This function is kept for backward compatibility but not used in GPU version
        G = nx.Graph()
        G.add_nodes_from(range(data.num_nodes))
        
        edge_index = data.edge_index.cpu().numpy()
        for i in range(edge_index.shape[1]):
            src, dst = edge_index[0, i], edge_index[1, i]
            G.add_edge(int(src), int(dst))
        
        return G

