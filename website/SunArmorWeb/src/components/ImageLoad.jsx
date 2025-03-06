import React, { useState, useRef } from 'react'

function ImageUploader() {
    const [selectedFile, setSelectedFile] = useState(null)
    const [preview, setPreview] = useState(null)
    const [uploading, setUploading] = useState(false)
    const [response, setResponse] = useState(null)
    const [error, setError] = useState(null)

    const imgInput = useRef(null)

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
        <div className='w-75 sm:w-100 mx-auto p-6 bg-neutral-700 rounded-lg shadow-lg'>
            <div className='mb-4'>
                <input
                    type='file'
                    accept='image/*'
                    onChange={handleFileChange}
                    style={{ display: 'none' }}
                    ref={imgInput}
                />
                <button
                    onClick={() => imgInput.current.click()}
                    className='w-full p-2 rounded text-black bg-peach hover:bg-peach-light font-display'
                >
                    SELECT IMAGE
                </button>
            </div>

            {preview && (
                <div className='mb-4 flex flex-col items-center'>
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
                className={`w-full p-2 rounded text-black font-display ${
                    !selectedFile || uploading
                        ? 'bg-neutral-400'
                        : 'bg-peach hover:bg-peach-light'
                }`}
            >
                {uploading ? 'UPLOADING...' : 'UPLOAD'}
            </button>

            {error && (
                <div className='mt-4 p-3 bg-neutral-300 text-red-500 rounded font-display'>
                    ERROR: {error}
                </div>
            )}

            {response && (
                <div className='mt-4 p-3 bg-neutral-300 text-green-500 rounded font-display'>
                    <span>Success! {response.message}</span>
                    <span>
                        Label: {response.label}
                        <br />
                        Score: {response.predicted_class_score}
                        <br />
                        Result: {response.result}
                    </span>
                </div>
            )}
        </div>
    )
}

export default ImageUploader
