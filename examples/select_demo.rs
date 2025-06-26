/* 
use cant::central::*;

fn main() {
    println!("Select Operation Demo - Embedding Layer Example");
    
    // Create a simple embedding matrix with vocabulary size 5 and embedding dimension 3
    // Each row represents the embedding for a token
    let embedding_matrix = Tensor::create_tensor_data_and_shape_and_operation(
        Shape::new(vec![5, 3]),
        vec![
            // Token 0: [1.0, 2.0, 3.0]
            1.0, 2.0, 3.0,
            // Token 1: [4.0, 5.0, 6.0]
            4.0, 5.0, 6.0,
            // Token 2: [7.0, 8.0, 9.0]
            7.0, 8.0, 9.0,
            // Token 3: [10.0, 11.0, 12.0]
            10.0, 11.0, 12.0,
            // Token 4: [13.0, 14.0, 15.0]
            13.0, 14.0, 15.0,
        ],
        Operation::Nop,
    );
    
    println!("Embedding matrix shape: {:?}", embedding_matrix.shape.dimensions());
    println!("Embedding matrix data:");
    let embedding_data = embedding_matrix.item();
    for i in 0..5 {
        println!("  Token {}: [{:.1}, {:.1}, {:.1}]", 
                 i, embedding_data[[i, 0]], embedding_data[[i, 1]], embedding_data[[i, 2]]);
    }
    
    // Create indices tensor - selecting tokens [1, 3, 0, 2]
    let indices = Tensor::create_tensor_data_and_shape_and_operation(
        Shape::new(vec![4]),
        vec![1.0, 3.0, 0.0, 2.0],
        Operation::Nop,
    );
    
    println!("\nSelecting tokens with indices: [1, 3, 0, 2]");
    
    // Method 1: Using the select method directly
    let result_direct = embedding_matrix.select(indices.id);
    println!("\nUsing select() method:");
    println!("Result shape: {:?}", result_direct.shape.dimensions());
    
    let result_data = result_direct.item();
    for i in 0..4 {
        println!("  Selected row {}: [{:.1}, {:.1}, {:.1}]", 
                 i, result_data[[i, 0]], result_data[[i, 1]], result_data[[i, 2]]);
    }
    
    // Method 2: Using the view method with tensor indexing (PyTorch-like)
    let result_view = embedding_matrix.view(Indexable::FromTensor(indices.id));
    println!("\nUsing view() method (PyTorch-like interface):");
    println!("Result shape: {:?}", result_view.shape.dimensions());
    
    let view_data = result_view.item();
    for i in 0..4 {
        println!("  Selected row {}: [{:.1}, {:.1}, {:.1}]", 
                 i, view_data[[i, 0]], view_data[[i, 1]], view_data[[i, 2]]);
    }
    
    // Verify both methods produce the same results
    println!("\nVerifying both methods produce identical results...");
    let mut same = true;
    for i in 0..4 {
        for j in 0..3 {
            if (result_data[[i, j]] - view_data[[i, j]]).abs() > 1e-6 {
                same = false;
                break;
            }
        }
        if !same { break; }
    }
    
    if same {
        println!("✓ Both methods produce identical results!");
    } else {
        println!("✗ Methods produce different results!");
    }
    
    println!("\nSelect operation successfully implemented!");
    println!("This enables PyTorch-like embedding layer functionality in Cant ML library.");
}

*/

fn main() {
    
}