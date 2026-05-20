def fuse_risk(audio_result, video_result):
    audio_risk = audio_result["riskLevel"]
    video_risk = video_result["visualRisk"]

    if audio_risk == "URGENTE":
        final_risk = "URGENTE"

    elif audio_risk == "MONITORAR" and video_risk == "HIGH":
        final_risk = "URGENTE"

    elif video_risk == "HIGH":
        final_risk = "MONITORAR"

    else:
        final_risk = "ROTINA"

    human_review = final_risk in ["URGENTE", "MONITORAR"]

    return {
        "audioRisk": audio_risk,
        "videoRisk": video_risk,
        "finalRisk": final_risk,
        "humanReviewRequired": human_review
    }


if __name__ == "__main__":
    audio_result = {
        "riskLevel": "MONITORAR"
    }

    video_result = {
        "visualRisk": "HIGH"
    }

    result = fuse_risk(audio_result, video_result)

    print(result)