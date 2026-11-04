const { onDocumentCreated } = require("firebase-functions/v2/firestore");
const { initializeApp } = require("firebase-admin/app");
const {
  getFirestore,
  FieldValue,
} = require("firebase-admin/firestore");
const { getMessaging } = require("firebase-admin/messaging");

initializeApp();

exports.sendPeerGridNotification = onDocumentCreated(
  "notifications/{notificationId}",
  async (event) => {
    const notification = event.data?.data();

    if (!notification) {
      return;
    }

    const recipientId = String(notification.recipientId || "");

    if (!recipientId) {
      return;
    }

    // Find the recipient's FCM tokens.
    const userSnap = await getFirestore()
      .collection("users")
      .doc(recipientId)
      .get();

    if (!userSnap.exists) {
      return;
    }

    const userData = userSnap.data() || {};

    const tokens = Array.isArray(userData.fcmTokens)
      ? userData.fcmTokens.filter(
          (token) => typeof token === "string" && token.length > 0
        )
      : [];

    if (tokens.length === 0) {
      return;
    }

    const title = String(notification.title || "PeerGrid");

    const body = String(
      notification.body ||
        "You have a new PeerGrid notification."
    );

    const type = String(notification.type || "");

    const relatedId = String(
      notification.relatedId || ""
    );

    const senderId = String(
      notification.senderId || ""
    );

    const senderName = String(
      notification.senderName || "Student"
    );

    // Send the notification to all registered devices.
    const response = await getMessaging().sendEachForMulticast({
      tokens: tokens,

      notification: {
        title: title,
        body: body,
      },

      data: {
        type: type,
        relatedId: relatedId,
        senderId: senderId,
        senderName: senderName,
        senderProfileImage: String(
          notification.senderProfileImage || ""
        ),
      },

      android: {
        priority: "high",

        notification: {
          sound: "default",
        },
      },
    });

    // Remove invalid/expired FCM tokens.
    const invalidTokens = [];

    response.responses.forEach((result, index) => {
      const code = result.error?.code || "";

      if (
        code.includes(
          "registration-token-not-registered"
        ) ||
        code.includes(
          "invalid-registration-token"
        )
      ) {
        invalidTokens.push(tokens[index]);
      }
    });

    if (invalidTokens.length > 0) {
      await getFirestore()
        .collection("users")
        .doc(recipientId)
        .update({
          fcmTokens: FieldValue.arrayRemove(
            ...invalidTokens
          ),
        });
    }
  }
);