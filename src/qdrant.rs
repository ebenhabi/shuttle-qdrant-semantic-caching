use anyhow::Result;
use std::env;
use std::path::PathBuf;
use std::sync::Arc;

use qdrant_client::prelude::{CreateCollection, Distance, PointStruct, QdrantClient};
use qdrant_client::qdrant::{
    vectors_config::Config,
    with_payload_selector::SelectorOptions,
    VectorParams,
    VectorsConfig,
    WithPayloadSelector,
    SearchPoints
};
use async_openai::types::{
    ChatCompletionRequestMessage,
    ChatCompletionRequestSystemMessageArgs,
    ChatCompletionRequestUserMessageArgs,
    CreateChatCompletionRequestArgs,
    CreateEmbeddingRequest,
    EmbeddingInput
};

use async_openai::{config::OpenAIConfig, Client, Embeddings};

#[derive(Clone)]
pub struct RAGSystem {
    qdrant_client: Arc<QdrantClient>,
    openai_client: Client<OpenAIConfig>
}

static REGULAR_COLLECTION_NAME: &str = "my-collection";
static CACHE_COLLECTION_NAME: &str = "my-collection_cached";

impl RAGSystem {
    // Initialising our regular collection
    pub fn new(qdrant_client: QdrantClient) -> Self {
        let openai_api_key = env::var("OPENAI_API_KEY").unwrap();

        let openai_config = OpenAIConfig::new()
            .with_api_key(openai_api_key)
            .with_org_id("qdrant-shuttle-semantic-cache");

        let openai_client = Client::with_config(openai_config);

        Self {
            openai_client,
            qdrant_client: Arc::new(qdrant_client),
        }
    }

    /* Creating collection */
    pub async fn create_regular_collection(&self) -> Result<()> {
        self.qdrant_client
            .create_collection(&CreateCollection {
                collection_name: REGULAR_COLLECTION_NAME.to_string(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: 1536,
                        distance: Distance::Cosine.into(),
                        hnsw_config: None,
                        quantization_config: None,
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;

        Ok(())
    }

    /* Creating cached collection */
    pub async fn create_cache_collection(&self) -> Result<()> {
        self.qdrant_client
            .create_collection(&CreateCollection {
                collection_name: CACHE_COLLECTION_NAME.to_string(),
                vectors_config: Some(VectorsConfig {
                    config: Some(Config::Params(VectorParams {
                        size: 1536,
                        distance: Distance::Euclid.into(),
                        hnsw_config: None,
                        quantization_config: None,
                        on_disk: None,
                        ..Default::default()
                    })),
                }),
                ..Default::default()
            })
            .await?;

        Ok(())
    }

    /* Creating embeddings */
    pub async fn embed_and_upsert_csv_file(&self, file_path: PathBuf) -> Result<()> {
        let file_contents = std::fs::read_to_string(&file_path)?;

        // note here that we skip 1 because CSV files typically have headers
        // if you don't have any headers, you can remove it
        let chunked_file_contents: Vec<String> = file_contents
            .lines()
            .skip(1)
            .map(|x| x.to_owned())
            .collect();

        let embedding_request = CreateEmbeddingRequest {
            model: "text-embedding-ada-002".to_string(),
            input: EmbeddingInput::StringArray(chunked_file_contents.to_owned()),
            encoding_format: None,
            user: None,
            dimensions: Some(1536),
        };

        let embeddings = Embeddings::new(&self.openai_client)
            .create(embedding_request)
            .await?;

        if embeddings.data.is_empty() {
            return Err(anyhow::anyhow!(
                "There were no embeddings returned by OpenAI"
            ));
        }

        let embeddings_vec: Vec<Vec<f32>> =
            embeddings.data.into_iter().map(|x| x.embedding).collect();

        // note that we create the upsert_embedding function later on
        for embedding in embeddings_vec {
            self.upset_embedding(embedding, file_contents.clone())
                .await?;
        }

        Ok(())
    }

    /* need to embed any further inputs to search for any matching embeddings */
    pub async fn embed_prompt(&self, prompt: &str) -> Result<Vec<f32>> {
        let embedding_request = CreateEmbeddingRequest {
            model: "text-embedding-ada-002".to_string(),
            input: EmbeddingInput::String(prompt.to_owned()),
            encoding_format: None,
            user: None,
            dimensions: Some(1536),
        };

        let embeddings = Embeddings::new(&self.openai_client)
            .create(embedding_request)
            .await?;

        if embeddings.data.is_empty() {
            return Err(anyhow::anyhow!(
                "There were no embeddings returned by OpenAI!"
            ));
        }

        Ok(embeddings.data.into_iter().next().unwrap().embedding)
    }

    /* Upserting embeddings */
    async fn upset_embedding(&self, embedding: Vec<f32>, file_contents: String) -> Result<()> {
        let playload = serde_json::json!({
            "document": file_contents
        })
            .try_into()
            .map_err(|x| anyhow::anyhow!("Ran into an error when converting the payload: {x}"))?;

        let points = vec![PointStruct::new(
            uuid::Uuid::new_v4().to_string(),
            embedding,
            playload,
        )];

        self.qdrant_client
            .upsert_points(REGULAR_COLLECTION_NAME.to_owned(), None, points, None)
            .await?;

        Ok(())
    }

    /* Adding things to our cache */
    pub async fn add_to_cache(&self, embedding: Vec<f32>, answer: &str) -> Result<()> {
        let payload = serde_json::json!({
            "answer": answer
        })
            .try_into()
            .map_err(|x| anyhow::anyhow!("Ran into an error when converting the playload: {x}"))?;

        let points = vec![PointStruct::new(
            uuid::Uuid::new_v4().to_string(),
            embedding,
            payload,
        )];

        self.qdrant_client
            .upsert_points(CACHE_COLLECTION_NAME.to_owned(), None, points, None)
            .await?;

        Ok(())
    }

    /* Searching Qdrant collections */
    pub async fn search(&self, embedding: Vec<f32>) -> Result<String> {
        let payload_selector = WithPayloadSelector {
            selector_options: Some(SelectorOptions::Enable(true)),
        };

        let search_points = SearchPoints {
            collection_name: REGULAR_COLLECTION_NAME.to_string(),
            vector: embedding,
            limit: 1,
            with_payload: Some(payload_selector),
            ..Default::default()
        };

        let search_result = self
            .qdrant_client
            .search_points(&search_points)
            .await
            .inspect_err(|x| println!("An error occurred while searching for points: {x}"))?;

        let result = search_result.result.into_iter().next();

        let Some(result) = result else {
            return Err(anyhow::anyhow!("There's nothing matching."))
        };

        Ok(result.payload.get("document").unwrap().to_string())
    }

    /* searching your cache collection */
    pub async fn search_cache(&self, embedding: Vec<f32>) -> Result<String> {
        let payload_selector = WithPayloadSelector {
            selector_options: Some(SelectorOptions::Enable(true)),
        };

        let search_points = SearchPoints {
            collection_name: CACHE_COLLECTION_NAME.to_string(),
            vector: embedding,
            limit: 1,
            with_payload: Some(payload_selector),
            ..Default::default()
        };

        let search_result = self
            .qdrant_client
            .search_points(&search_points)
            .await
            .inspect_err(|x| println!("An error occurred while searching for points: {x}"))?;

        let result = search_result.result.into_iter().next();

        let Some(result) = result else {
            return Err(anyhow::anyhow!("There's nothing matching."))
        };

        Ok(result.payload.get("answer").unwrap().to_string())
    }

    /* Prompting */
    pub async fn prompt(&self, prompt: &str, context: &str) -> Result<String> {
        let input = format!(
            "{prompt}\
            \
            Provided context:\
            {context}
            "
        );

        let res = self.openai_client
            .chat()
            .create(
                CreateChatCompletionRequestArgs::default()
                    .model("gpt-4o")
                    .messages(vec![
                        // First we add the system message to define what the Agent does
                        ChatCompletionRequestMessage::System(
                            ChatCompletionRequestSystemMessageArgs::default()
                                .build()?,
                        ),
                        // Then we add our prompt
                        ChatCompletionRequestMessage::User(
                            ChatCompletionRequestUserMessageArgs::default()
                                .content(input)
                                .build()?,
                        ),
                    ])
                    .build()?,
            )
            .await
            .map(|res| {
                // We extract the first one
                match res.choices[0].message.content.clone() {
                    Some(res) => Ok(res),
                    None => Err(anyhow::anyhow!("There was no result from OpenAI")),
                }
            })??;

        println!("Retrieved result from prompt: {res}");

        Ok(res)
    }
}