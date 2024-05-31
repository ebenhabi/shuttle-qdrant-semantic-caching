mod qdrant;

use std::env;

use axum::{Json, extract::State, response::IntoResponse, http::StatusCode};
use axum::{routing::post, Router};

use qdrant::RAGSystem;
use qdrant_client::client::QdrantClient;

use serde::Deserialize;
use shuttle_runtime::SecretStore;

/* Using Qdrant in a Rust web service */
#[derive(Deserialize)]
struct Prompt {
    prompt: String,
}

async fn prompt(
    State(state): State<RAGSystem>,
    Json(prompt): Json<Prompt>,
) -> Result<impl IntoResponse, impl IntoResponse> {
    let embedding = match state.embed_prompt(&prompt.prompt).await {
        Ok(embedding) => embedding,
        Err(e) => {
            return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("An error occurred while embedding the prompt: {e}"),
                ))
        }
    };

    if let Ok(answer) = state.search_cache(embedding.clone()).await {
        return Ok(answer);
    }

    let search_result = match state.search(embedding.clone()).await {
        Ok(res) => res,
        Err(e) => {
            return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("An error occurred while prompting: {e}"),
                ))
        }
    };

    let llm_response = match state.prompt(&prompt.prompt, &search_result).await {
        Ok(prompt_result) => prompt_result,
        Err(e) => {
            return Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    format!("Something went wrong while prompting: {e}"),
                ))
        }
    };

    if let Err(e) = state.add_to_cache(embedding, &llm_response).await {
        return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                format!("Something went wrong while adding item to the cache: {e}"),
            ));
    };

    Ok(llm_response)
}

#[shuttle_runtime::main]
async fn main(
    #[shuttle_qdrant::Qdrant(
        cloud_url = "{secrets.QDRANT_URL}",
        api_key = "{secrets.QDRANT_API_TOKEN}"
    )]
    qdrant_client: QdrantClient,
    #[shuttle_runtime::Secrets] secrets: SecretStore,
) -> shuttle_axum::ShuttleAxum {
    secrets.into_iter().for_each(|x| env::set_var(x.0, x.1));

    let rag = RAGSystem::new(qdrant_client);

    let setup_required = true;

    if setup_required {
        rag.create_cache_collection().await?;
        rag.create_cache_collection().await?;

        rag.embed_and_upsert_csv_file("text.csv".into()).await?;
    }

    let router = Router::new().route("/prompt", post(prompt)).with_state(rag);

    Ok(router.into())
}
