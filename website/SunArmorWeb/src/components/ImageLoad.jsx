import React, { useState } from 'react'

function ImageUploader() {
    const [selectedFile, setSelectedFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [uploading, setUploading] = useState(false)
    const [response, setResponse] = useState(null)
    const [error, setError] = useState(null)

    // Replace with your ngrok URL from Colab
    const API_URL = 'https://bcfb-130-86-97-245.ngrok-free.app/upload'

    const handleFileChange = (event) => {
        const file = event.target.files[0]

        if (file) {
            setSelectedFile(file)

            // Create preview
            const reader = new FileReader()
            reader.onloadend = () => {
                setPreview(reader.result)
            }
            reader.readAsDataURL(file)

            // Reset states
            setResponse(null)
            setError(null)
        }
    }

    const handleUpload = async () => {
        if (!selectedFile) {
            setError('Please select an image first')
            return
        }

        setUploading(true)
        setError(null)

        try {
            const formData = new FormData()
            formData.append('image', selectedFile)

            const response = await fetch(API_URL, {
                method: 'POST',
                body: formData,
                // No need to set Content-Type header, as FormData sets it automatically with boundary
            })

            // Check Content-Type before parsing JSON
            const contentType = response.headers.get('content-type')
            if (!contentType || !contentType.includes('application/json')) {
                throw new Error('Server returned invalid response')
            }

            const data = await response.json()
            console.log(data)

            if (!response.ok) {
                throw new Error(data.error || 'Upload failed')
            }

            setResponse(data)
        } catch (err) {
            setError(err.message || 'Error uploading image')
            console.error('Upload error:', err)
        } finally {
            setUploading(false)
        }
    }

    return (
        <div className='max-w-md mx-auto p-6 bg-white rounded-lg shadow-md'>
            <h2 className='text-2xl font-bold mb-4 text-black'>
                Skin Cancer Detection
            </h2>

            <div className='mb-4'>
                <label className='block text-black mb-2'>Select Image</label>
                <input
                    type='file'
                    accept='image/*'
                    onChange={handleFileChange}
                    className='w-full p-2 rounded-full bg-green-500'
                />
            </div>

            {preview && (
                <div className='mb-4'>
                    <p className='text-gray-700 mb-2'>Preview:</p>
                    <img
                        src={preview}
                        alt='Preview'
                        className='max-h-64 max-w-full rounded'
                    />
                </div>
            )}

            <button
                onClick={handleUpload}
                disabled={!selectedFile || uploading}
                className={`w-full p-2 rounded text-white font-bold ${
                    !selectedFile || uploading
                        ? 'bg-gray-400'
                        : 'bg-blue-500 hover:bg-blue-600'
                }`}
            >
                {uploading ? 'Uploading...' : 'Upload to Colab'}
            </button>

            {error && (
                <div className='mt-4 p-3 bg-red-100 text-red-700 rounded'>
                    Error: {error}
                </div>
            )}

            {response && (
                <div className='mt-4 p-3 bg-green-100 text-green-700 rounded'>
                    <p>
                        <strong>Success!</strong> {response.message}
                    </p>
                    <p>
                        Label: {response.label}
                        <br />
                        Score: {response.predicted_class_score}
                        <br />
                        Result: {response.result}
                    </p>
                </div>
            )}
        </div>
    )
}

export default ImageUploader
