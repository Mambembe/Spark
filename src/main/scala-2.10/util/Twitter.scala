package util

import twitter4j.auth.{Authorization, OAuthAuthorization}
import twitter4j.conf.ConfigurationBuilder

object Twitter {

  def auth(consumerKey: String, consumerSecret: String, accessToken: String, accessTokenSecret: String): Option[Authorization] = {
    val c =
      new ConfigurationBuilder()
        .setOAuthConsumerKey("aPYODVLNmwLsVfpf9JnrnFXwZ")
        .setOAuthConsumerSecret("VZge3rcIhpgaF9uhzm9H80pTg6ireTCc6IL2MhLj8iwTOR7hGs")
        .setOAuthAccessToken("775822357-QRrUK7VXFdaY2JprL9pHxfIqI9euQF6tsfZKlXiu")
        .setOAuthAccessTokenSecret("nrQ717sVgBPEA5AZFhu9Ne7OgAZsq6wSwnFSC88Jusd3l")
        .build()
    Option(new OAuthAuthorization(c))
  }

}
